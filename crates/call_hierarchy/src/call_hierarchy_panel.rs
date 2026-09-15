use std::{collections::HashMap, collections::HashSet, ops::Range, sync::Arc};

use editor::{Editor, SelectionEffects, scroll::Autoscroll};
use fs::Fs;
use gpui::{
    App, AsyncWindowContext, Bounds, Context, Entity, EntityId, EventEmitter, FocusHandle,
    Focusable, ListHorizontalSizingBehavior, ListSizingBehavior, ParentElement, Render, Styled,
    Task, UniformListScrollHandle, WeakEntity, Window, actions, point, size, uniform_list,
};
use language::{Anchor, ToPoint};
use project::{CallHierarchyItem, Project};
use settings::{DockSide, Settings as _};
use ui::{Disclosure, ListItem, ListItemSpacing, Tooltip, prelude::*};
use workspace::{
    Workspace,
    dock::{DockPosition, Panel, PanelEvent},
};

use crate::{Call, CallHierarchyMode, CallHierarchySettings, fetch_calls};

const CALL_HIERARCHY_PANEL_KEY: &str = "CallHierarchyPanel";

/// Matches the project panel's default `indent_size`, so nesting reads the same across
/// Zed's trees (`ListItem`'s own default is narrower, at 12px).
const INDENT_STEP: f32 = 20.;
const DISCLOSURE_WIDTH: f32 = 16.;
/// Puts each guide under the centre of its parent row's disclosure: `ListItem`'s own
/// horizontal padding (`DynamicSpacing::Base06`, 6px at the default density) plus
/// half the disclosure column. Without this they are drawn hard against x = 0.
const INDENT_GUIDE_LEFT_OFFSET: Pixels = px(6. + DISCLOSURE_WIDTH / 2.);

actions!(
    call_hierarchy,
    [
        /// Toggles focus on the call hierarchy panel.
        ToggleFocus,
    ]
);

/// Lives in `zed_actions` rather than here so the editor's mouse context menu can
/// reference it without depending on this crate.
pub use zed_actions::ShowTree;

/// Identifies a call hierarchy item across fetches so that expansion state and
/// recursion detection survive reloading a subtree. `CallHierarchyItem` itself
/// derives neither `Eq` nor `Hash`, and two language servers reporting the same
/// buffer range are the same symbol for our purposes, so the server id is left out.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NodeKey {
    buffer: EntityId,
    start: Anchor,
    end: Anchor,
}

impl NodeKey {
    fn for_item(item: &CallHierarchyItem) -> Self {
        Self {
            buffer: item.buffer.entity_id(),
            start: item.range.start,
            end: item.range.end,
        }
    }
}

enum ChildState {
    NotLoaded,
    Loading,
    Loaded(Vec<Node>),
}

struct Node {
    call: Call,
    key: NodeKey,
    children: ChildState,
}

impl Node {
    fn new(call: Call) -> Self {
        Self {
            key: NodeKey::for_item(&call.item),
            call,
            children: ChildState::NotLoaded,
        }
    }
}

/// A visible row, produced by flattening the expanded parts of the tree.
#[derive(Clone)]
struct FlatEntry {
    path: Vec<usize>,
    depth: usize,
    expanded: bool,
    /// Set when this node already appears in its own ancestor chain. Expanding it
    /// would recurse forever, so it is rendered as a leaf with a marker instead.
    recursive: bool,
    /// Waiting on the language server. Shown inline on this row rather than as a
    /// placeholder child row, which would reflow everything below it twice - once
    /// when the placeholder appears and again when the real calls replace it.
    loading: bool,
}

#[derive(Default, PartialEq)]
enum PanelState {
    #[default]
    Empty,
    Loading,
    NoSymbol,
    Ready,
}


pub fn init(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(|workspace, _: &ToggleFocus, window, cx| {
            workspace.toggle_panel_focus::<CallHierarchyPanel>(window, cx);
        });
        workspace.register_action(|workspace, _: &ShowTree, window, cx| {
            let Some(panel) = workspace.panel::<CallHierarchyPanel>(cx) else {
                return;
            };
            let editor = workspace
                .active_item(cx)
                .and_then(|item| item.act_as::<Editor>(cx));
            workspace.open_panel::<CallHierarchyPanel>(window, cx);
            panel.update(cx, |panel, cx| match editor {
                Some(editor) => panel.show_for_editor(editor, window, cx),
                None => panel.set_no_symbol(cx),
            });
        });
    })
    .detach();
}

pub struct CallHierarchyPanel {
    workspace: WeakEntity<Workspace>,
    project: Entity<Project>,
    focus_handle: FocusHandle,
    mode: CallHierarchyMode,
    root: Option<CallHierarchyItem>,
    nodes: Vec<Node>,
    entries: Vec<FlatEntry>,
    /// Keyed by position in the tree, not by `NodeKey`: one function can appear at
    /// several places in the hierarchy, and expanding one occurrence must not expand
    /// or collapse the others.
    expanded: HashSet<Vec<usize>>,
    state: PanelState,
    fs: Arc<dyn Fs>,
    scroll_handle: UniformListScrollHandle,
    widest_entry_index: Option<usize>,
    root_task: Option<Task<()>>,
    child_tasks: HashMap<Vec<usize>, Task<()>>,
}

impl CallHierarchyPanel {
    pub async fn load(
        workspace: WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> anyhow::Result<Entity<Self>> {
        workspace.update_in(&mut cx, |workspace, window, cx| {
            Self::new(workspace, window, cx)
        })
    }

    fn new(
        workspace: &mut Workspace,
        _window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<Self> {
        let project = workspace.project().clone();
        let fs = project.read(cx).fs().clone();
        let workspace_handle = cx.entity().downgrade();
        cx.new(|cx| Self {
            workspace: workspace_handle,
            project,
            fs,
            focus_handle: cx.focus_handle(),
            mode: CallHierarchyMode::Incoming,
            root: None,
            nodes: Vec::new(),
            entries: Vec::new(),
            expanded: HashSet::default(),
            state: PanelState::Empty,
            scroll_handle: UniformListScrollHandle::default(),
            widest_entry_index: None,
            root_task: None,
            child_tasks: HashMap::default(),
        })
    }

    /// Seeds the tree from the symbol under the cursor, using the same
    /// `prepare_call_hierarchy` path the modal picker uses.
    ///
    /// The editor is passed in rather than resolved from `self.workspace`: callers
    /// reach this from a `Workspace` action handler, where the workspace entity is
    /// already leased for update, and reading it again panics.
    pub fn show_for_editor(
        &mut self,
        editor: Entity<Editor>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let editor_anchor = editor.update(cx, |editor, cx| {
            let selection = editor.selections.newest_anchor().head();
            editor
                .buffer()
                .read(cx)
                .text_anchor_for_position(selection, cx)
        });
        let Some((buffer, position)) = editor_anchor else {
            self.set_no_symbol(cx);
            return;
        };

        self.state = PanelState::Loading;
        self.nodes.clear();
        self.entries.clear();
        self.expanded.clear();
        self.child_tasks.clear();
        cx.notify();

        let prepare_task = self.project.update(cx, |project, cx| {
            project.prepare_call_hierarchy(&buffer, position, cx)
        });
        let project = self.project.clone();
        let mode = self.mode;

        self.root_task = Some(cx.spawn_in(window, async move |panel, cx| {
            let root_item = match prepare_task.await {
                Ok(items) => items.unwrap_or_default().into_iter().next(),
                Err(error) => {
                    log::error!("failed to prepare call hierarchy: {error:#}");
                    panel
                        .update(cx, |panel, cx| {
                            panel.set_no_symbol(cx);
                        })
                        .ok();
                    return;
                }
            };
            let Some(root_item) = root_item else {
                panel
                    .update(cx, |panel, cx| {
                        panel.set_no_symbol(cx);
                    })
                    .ok();
                return;
            };

            let calls = fetch_calls(&root_item, &project, mode, cx).await;
            let root_call = crate::root_call(&root_item, &project, cx).await;
            panel
                .update_in(cx, |panel, window, cx| {
                    panel.root = Some(root_item);
                    panel.set_root_calls(root_call, calls, window, cx);
                })
                .ok();
        }));
    }

    /// The root is an ordinary node at the top of the tree, so it renders like every
    /// other row - disclosure, signature, click-to-open - instead of a bespoke header.
    fn set_root_calls(
        &mut self,
        root_call: Call,
        calls: Vec<Call>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let mut root_node = Node::new(root_call);
        root_node.children = ChildState::Loaded(calls.into_iter().map(Node::new).collect());
        self.nodes = vec![root_node];
        self.expanded.insert(vec![0]);
        self.state = PanelState::Ready;
        self.rebuild_entries();
        self.auto_expand_first_level(window, cx);
        cx.notify();
    }

    /// Expands the first level up front so empty branches are visible without
    /// clicking each row - LSP offers no "has children" hint, so the only way to know
    /// is to ask. Capped because a widely-called function costs one request per
    /// caller, and that fan-out would queue ahead of hover and completions.
    fn auto_expand_first_level(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let limit = CallHierarchySettings::get_global(cx).auto_expand_limit;
        let child_count = match self.nodes.first().map(|root| &root.children) {
            Some(ChildState::Loaded(children)) => children.len(),
            _ => return,
        };
        if limit == 0 || child_count > limit {
            return;
        }
        for index in 0..child_count {
            let path = vec![0, index];
            let unloaded = Self::node_at(&self.nodes, &path)
                .is_some_and(|node| matches!(node.children, ChildState::NotLoaded));
            if unloaded {
                self.expanded.insert(path.clone());
                self.load_children(path, window, cx);
            }
        }
    }

    fn set_no_symbol(&mut self, cx: &mut Context<Self>) {
        self.state = PanelState::NoSymbol;
        self.root = None;
        self.nodes.clear();
        self.entries.clear();
        cx.notify();
    }

    fn set_mode(&mut self, mode: CallHierarchyMode, window: &mut Window, cx: &mut Context<Self>) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        let Some(root) = self.root.clone() else {
            cx.notify();
            return;
        };

        self.state = PanelState::Loading;
        self.nodes.clear();
        self.entries.clear();
        self.expanded.clear();
        self.child_tasks.clear();
        cx.notify();

        let project = self.project.clone();
        self.root_task = Some(cx.spawn_in(window, async move |panel, cx| {
            let calls = fetch_calls(&root, &project, mode, cx).await;
            let root_call = crate::root_call(&root, &project, cx).await;
            panel
                .update_in(cx, |panel, window, cx| {
                    panel.set_root_calls(root_call, calls, window, cx);
                })
                .ok();
        }));
    }

    fn node_at<'a>(nodes: &'a [Node], path: &[usize]) -> Option<&'a Node> {
        let (first, rest) = path.split_first()?;
        let node = nodes.get(*first)?;
        if rest.is_empty() {
            return Some(node);
        }
        match &node.children {
            ChildState::Loaded(children) => Self::node_at(children, rest),
            _ => None,
        }
    }

    fn node_at_mut<'a>(nodes: &'a mut [Node], path: &[usize]) -> Option<&'a mut Node> {
        let (first, rest) = path.split_first()?;
        let node = nodes.get_mut(*first)?;
        if rest.is_empty() {
            return Some(node);
        }
        match &mut node.children {
            ChildState::Loaded(children) => Self::node_at_mut(children, rest),
            _ => None,
        }
    }

    fn rebuild_entries(&mut self) {
        let mut entries = Vec::new();
        let mut path = Vec::new();
        let mut ancestors = HashSet::default();
        Self::flatten(
            &self.nodes,
            &mut path,
            0,
            &self.expanded,
            &mut ancestors,
            &mut entries,
        );
        self.entries = entries;
        // `uniform_list` sizes itself from a single item, so point it at the longest
        // row or the rest would be clipped instead of scrolling.
        self.widest_entry_index = self
            .entries
            .iter()
            .enumerate()
            .max_by_key(|(_, entry)| {
                let text_len = Self::node_at(&self.nodes, &entry.path)
                    .map(|node| {
                        node.call
                            .display
                            .label_text
                            .as_ref()
                            .map_or(node.call.display.name.len(), |label| label.len())
                    })
                    .unwrap_or_default();
                entry.depth * INDENT_STEP as usize + text_len
            })
            .map(|(index, _)| index);
    }

    fn flatten(
        nodes: &[Node],
        path: &mut Vec<usize>,
        depth: usize,
        expanded: &HashSet<Vec<usize>>,
        ancestors: &mut HashSet<NodeKey>,
        out: &mut Vec<FlatEntry>,
    ) {
        for (index, node) in nodes.iter().enumerate() {
            path.push(index);
            let recursive = ancestors.contains(&node.key);
            let is_expanded = !recursive && expanded.contains(path.as_slice());
            out.push(FlatEntry {
                path: path.clone(),
                depth,
                expanded: is_expanded,
                recursive,
                loading: matches!(node.children, ChildState::Loading),
            });
            if is_expanded
                && let ChildState::Loaded(children) = &node.children
            {
                let newly_inserted = ancestors.insert(node.key.clone());
                Self::flatten(children, path, depth + 1, expanded, ancestors, out);
                if newly_inserted {
                    ancestors.remove(&node.key);
                }
            }
            path.pop();
        }
    }

    fn toggle_expanded(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(entry) = self.entries.get(index).cloned() else {
            return;
        };
        if entry.recursive {
            return;
        }
        if self.expanded.remove(&entry.path) {
            self.rebuild_entries();
            cx.notify();
            return;
        }

        self.expanded.insert(entry.path.clone());
        let needs_load = Self::node_at(&self.nodes, &entry.path)
            .is_some_and(|node| matches!(node.children, ChildState::NotLoaded));
        if needs_load {
            self.load_children(entry.path, window, cx);
        } else {
            self.rebuild_entries();
            cx.notify();
        }
    }

    fn load_children(&mut self, path: Vec<usize>, window: &mut Window, cx: &mut Context<Self>) {
        let Some(node) = Self::node_at_mut(&mut self.nodes, &path) else {
            return;
        };
        node.children = ChildState::Loading;
        let item = node.call.item.clone();
        let project = self.project.clone();
        let mode = self.mode;
        self.rebuild_entries();
        cx.notify();

        let task = cx.spawn_in(window, {
            let path = path.clone();
            async move |panel, cx| {
                let calls = fetch_calls(&item, &project, mode, cx).await;
                panel
                    .update(cx, |panel, cx| {
                        if let Some(node) = Self::node_at_mut(&mut panel.nodes, &path) {
                            node.children =
                                ChildState::Loaded(calls.into_iter().map(Node::new).collect());
                        }
                        panel.rebuild_entries();
                        cx.notify();
                    })
                    .ok();
            }
        });
        self.child_tasks.insert(path, task);
    }

    fn open_entry(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(entry) = self.entries.get(index) else {
            return;
        };
        let Some(node) = Self::node_at(&self.nodes, &entry.path) else {
            return;
        };
        let buffer = node.call.target.buffer.clone();
        let target = node.call.target.range.start;
        self.open_location(buffer, target, window, cx);
    }

    fn open_location(
        &mut self,
        buffer: Entity<language::Buffer>,
        target: Anchor,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.workspace
            .update(cx, |workspace, cx| {
                let position = target.to_point(&buffer.read(cx).snapshot());
                let pane = workspace.active_pane().clone();
                let editor = workspace.open_project_item::<Editor>(
                    Some(pane),
                    buffer,
                    true,
                    true,
                    true,
                    true,
                    window,
                    cx,
                );
                editor.update(cx, |editor, cx| {
                    editor.change_selections(
                        SelectionEffects::scroll(Autoscroll::center()),
                        window,
                        cx,
                        |selections| selections.select_ranges([position..position]),
                    );
                });
            })
            .ok();
    }

    fn render_entry(&self, index: usize, cx: &mut Context<Self>) -> Option<gpui::AnyElement> {
        let entry = self.entries.get(index)?;
        let node = Self::node_at(&self.nodes, &entry.path)?;
        let display = node.call.display.clone();
        let recursive = entry.recursive;

        let mut label = h_flex()
            .gap_1p5()
            .child(Label::new(display.name.clone()).single_line());
        if recursive {
            label = label.child(
                Label::new("recursive")
                    .color(Color::Warning)
                    .size(LabelSize::Small),
            );
        }
        if entry.loading {
            label = label.child(
                Label::new("loading…")
                    .color(Color::Muted)
                    .size(LabelSize::Small)
                    .italic(),
            );
        }

        let expanded = entry.expanded;
        let indent = px(entry.depth as f32 * INDENT_STEP);

        Some(
            ListItem::new(index)
                .spacing(ListItemSpacing::Sparse)
                .selectable(true)
                // Indent and disclosure are drawn as ordinary row content rather than
                // via `indent_level`/`toggle`. Those shift only `ListItem`'s inner
                // element while its hover background stays full width, and they place
                // the disclosure absolutely at `left: -1rem` - so the arrow lands
                // outside its own row's highlight and child rows highlight as if they
                // were at the root level.
                .start_slot(
                    h_flex()
                        .child(div().w(indent))
                        .child(div().w(px(DISCLOSURE_WIDTH)).children(
                            (!recursive).then(|| {
                                Disclosure::new(("disclosure", index), expanded).on_click(
                                    cx.listener(move |panel, _, window, cx| {
                                        panel.toggle_expanded(index, window, cx);
                                    }),
                                )
                            }),
                        )),
                )
                .on_click(cx.listener(move |panel, _, window, cx| {
                    panel.open_entry(index, window, cx);
                }))
                // The call site's file and line would crowd a narrow panel, so it lives
                // in the tooltip rather than on the row.
                .when_some(display.path, |this, location| {
                    this.tooltip(move |_, cx| Tooltip::simple(location.clone(), cx))
                })
                .child(label)
                .into_any_element(),
        )
    }

    fn render_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mode = self.mode;
        h_flex()
            .p_1()
            .gap_1()
            .justify_between()
            .border_b_1()
            .border_color(cx.theme().colors().border)
            // The inspected symbol is the tree's root row now, so the header only
            // carries the panel's name and the direction toggle.
            .child(Label::new("Call Hierarchy").single_line())
            .child(
                Button::new(
                    "toggle-call-hierarchy-direction",
                    match mode {
                        CallHierarchyMode::Incoming => "Incoming",
                        CallHierarchyMode::Outgoing => "Outgoing",
                    },
                )
                .label_size(LabelSize::Small)
                .on_click(cx.listener(move |panel, _, window, cx| {
                    panel.set_mode(mode.opposite(), window, cx);
                })),
            )
    }
}

impl Render for CallHierarchyPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entry_count = self.entries.len();
        v_flex()
            .key_context("CallHierarchyPanel")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|panel, _: &crate::ToggleDirection, window, cx| {
                panel.set_mode(panel.mode.opposite(), window, cx);
            }))
            .size_full()
            .child(self.render_header(cx))
            .map(|this| match self.state {
                PanelState::Empty => this.child(
                    div()
                        .p_2()
                        .child(Label::new("No call hierarchy loaded.").color(Color::Muted)),
                ),
                PanelState::Loading => this.child(
                    div()
                        .p_2()
                        .child(Label::new("Loading…").color(Color::Muted)),
                ),
                PanelState::NoSymbol => this.child(div().p_2().child(
                    Label::new("No symbol under the cursor.").color(Color::Muted),
                )),
                PanelState::Ready => this.child(
                    uniform_list(
                        "call-hierarchy-entries",
                        entry_count,
                        cx.processor(|panel, range: Range<usize>, _window, cx| {
                            range
                                .filter_map(|index| panel.render_entry(index, cx))
                                .collect()
                        }),
                    )
                    .size_full()
                    .track_scroll(&self.scroll_handle)
                    // Signatures are long and the panel is narrow, so let rows keep
                    // their full width and scroll sideways instead of truncating.
                    .with_sizing_behavior(ListSizingBehavior::Infer)
                    .with_horizontal_sizing_behavior(ListHorizontalSizingBehavior::Unconstrained)
                    .with_width_from_item(self.widest_entry_index)
                    .with_decoration(
                        ui::indent_guides(
                            px(INDENT_STEP),
                            ui::IndentGuideColors::panel(cx),
                        )
                        .with_compute_indents_fn(
                            cx.entity(),
                            |panel, range, _, _| {
                                panel
                                    .entries
                                    .get(range)
                                    .map(|entries| {
                                        entries.iter().map(|entry| entry.depth).collect()
                                    })
                                    .unwrap_or_default()
                            },
                        )
                        .with_render_fn(cx.entity(), |_, params, _, _| {
                            let indent_size = params.indent_size;
                            let item_height = params.item_height;
                            params
                                .indent_guides
                                .into_iter()
                                .map(|layout| ui::RenderedIndentGuide {
                                    bounds: Bounds::new(
                                        point(
                                            layout.offset.x * indent_size
                                                + INDENT_GUIDE_LEFT_OFFSET,
                                            layout.offset.y * item_height,
                                        ),
                                        size(px(1.), layout.length * item_height),
                                    ),
                                    layout,
                                    is_active: false,
                                    hitbox: None,
                                })
                                .collect()
                        }),
                    ),
                ),
            })
    }
}

impl Focusable for CallHierarchyPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<PanelEvent> for CallHierarchyPanel {}

impl Panel for CallHierarchyPanel {
    fn persistent_name() -> &'static str {
        "Call Hierarchy Panel"
    }

    fn panel_key() -> &'static str {
        CALL_HIERARCHY_PANEL_KEY
    }

    fn position(&self, _: &Window, cx: &App) -> DockPosition {
        match CallHierarchySettings::get_global(cx).dock {
            DockSide::Left => DockPosition::Left,
            DockSide::Right => DockPosition::Right,
        }
    }

    fn position_is_valid(&self, position: DockPosition) -> bool {
        matches!(position, DockPosition::Left | DockPosition::Right)
    }

    /// Writes through to settings rather than storing the side locally: the workspace
    /// only re-docks a panel when the settings store changes, so a local field leaves
    /// the panel sitting where it was.
    fn set_position(&mut self, position: DockPosition, _: &mut Window, cx: &mut Context<Self>) {
        settings::update_settings_file(self.fs.clone(), cx, move |settings, _| {
            let dock = match position {
                DockPosition::Left | DockPosition::Bottom => DockSide::Left,
                DockPosition::Right => DockSide::Right,
            };
            settings.call_hierarchy.get_or_insert_default().dock = Some(dock);
        });
    }

    fn default_size(&self, _: &Window, _: &App) -> Pixels {
        px(300.)
    }

    fn icon(&self, _: &Window, _: &App) -> Option<IconName> {
        Some(IconName::ListTree)
    }

    fn icon_tooltip(&self, _: &Window, _: &App) -> Option<&'static str> {
        Some("Call Hierarchy Panel")
    }

    fn toggle_action(&self) -> Box<dyn gpui::Action> {
        Box::new(ToggleFocus)
    }

    fn activation_priority(&self) -> u32 {
        7
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use language::Location;
    use project::FakeFs;
    use serde_json::json;
    use util::path;
    use workspace::AppState;

    #[gpui::test]
    async fn test_flatten_only_descends_into_expanded_nodes(cx: &mut TestAppContext) {
        let project = test_project(cx).await;
        let child = node(make_call("child", 5, &project, cx).await);
        let mut parent = node(make_call("parent", 1, &project, cx).await);
        let sibling = node(make_call("sibling", 9, &project, cx).await);
        parent.children = ChildState::Loaded(vec![child]);
        let nodes = vec![parent, sibling];

        let collapsed = flatten_for_test(&nodes, &HashSet::default());
        assert_eq!(collapsed.len(), 2);
        assert_eq!(collapsed[0].depth, 0);
        assert_eq!(collapsed[1].depth, 0);

        let expanded = HashSet::from_iter([vec![0]]);
        let expanded = flatten_for_test(&nodes, &expanded);
        assert_eq!(expanded.len(), 3);
        assert_eq!(expanded[0].depth, 0);
        assert_eq!(expanded[1].depth, 1, "child sits under its parent");
        assert_eq!(expanded[2].depth, 0, "sibling returns to the root level");
    }

    #[gpui::test]
    async fn test_flatten_marks_self_recursive_nodes(cx: &mut TestAppContext) {
        let project = test_project(cx).await;
        // Same name and line, so both nodes share a `NodeKey` the way a language
        // server reports a directly recursive function as its own caller.
        let inner = node(make_call("recurse", 20, &project, cx).await);
        let mut outer = node(make_call("recurse", 20, &project, cx).await);
        assert_eq!(inner.key, outer.key, "both nodes describe the same symbol");
        outer.children = ChildState::Loaded(vec![inner]);
        let nodes = vec![outer];

        let entries = flatten_for_test(&nodes, &HashSet::from_iter([vec![0]]));
        assert_eq!(entries.len(), 2);
        assert!(!entries[0].recursive, "the outer node is not yet recursive");
        assert!(
            entries[1].recursive,
            "a node repeating in its own ancestry is marked recursive"
        );
        assert!(
            !entries[1].expanded,
            "a recursive node stays collapsed so flattening terminates"
        );
    }

    #[gpui::test]
    async fn test_expansion_is_tracked_per_position_not_per_symbol(cx: &mut TestAppContext) {
        let project = test_project(cx).await;
        // One function commonly appears at several places in a call tree - here the
        // same `beta` calls two different functions. Expanding one occurrence must
        // leave the other alone.
        let mut first = node(make_call("beta", 5, &project, cx).await);
        let mut second = node(make_call("beta", 5, &project, cx).await);
        assert_eq!(first.key, second.key, "both rows are the same symbol");
        first.children =
            ChildState::Loaded(vec![node(make_call("main", 1, &project, cx).await)]);
        second.children =
            ChildState::Loaded(vec![node(make_call("main", 1, &project, cx).await)]);
        let nodes = vec![first, second];

        let entries = flatten_for_test(&nodes, &HashSet::from_iter([vec![0]]));
        assert_eq!(entries.len(), 3, "only the first occurrence expands");
        assert!(entries[0].expanded);
        assert_eq!(entries[1].depth, 1, "its child is shown");
        assert!(
            !entries[2].expanded,
            "the other occurrence of the same symbol stays collapsed"
        );
    }

    #[gpui::test]
    async fn test_flatten_marks_fetching_nodes_without_adding_a_row(cx: &mut TestAppContext) {
        let project = test_project(cx).await;
        let mut fetching = node(make_call("beta", 5, &project, cx).await);
        fetching.children = ChildState::Loading;
        let nodes = vec![fetching];

        let entries = flatten_for_test(&nodes, &HashSet::from_iter([vec![0]]));
        assert_eq!(
            entries.len(),
            1,
            "loading is shown on the row itself, so nothing is inserted below it"
        );
        assert!(entries[0].loading);
    }

    fn flatten_for_test(nodes: &[Node], expanded: &HashSet<Vec<usize>>) -> Vec<FlatEntry> {
        let mut entries = Vec::new();
        CallHierarchyPanel::flatten(
            nodes,
            &mut Vec::new(),
            0,
            expanded,
            &mut HashSet::default(),
            &mut entries,
        );
        entries
    }

    fn node(call: Call) -> Node {
        Node::new(call)
    }

    async fn test_project(cx: &mut TestAppContext) -> Entity<Project> {
        cx.update(|cx| {
            let _state = AppState::test(cx);
            editor::init(cx);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/test"),
            json!({"src": {"main.rs": "fn main() {}\n".repeat(40)}}),
        )
        .await;
        Project::test(fs, [path!("/test").as_ref()], cx).await
    }

    async fn make_call(
        name: &str,
        line: u32,
        project: &Entity<Project>,
        cx: &mut TestAppContext,
    ) -> Call {
        let buffer = project
            .update(cx, |project, cx| {
                project.open_local_buffer(std::path::Path::new(path!("/test/src/main.rs")), cx)
            })
            .await
            .unwrap();
        let (range, selection_range) = buffer.read_with(cx, |buffer, _| {
            (
                buffer.anchor_after(language::Point::new(line, 0))
                    ..buffer.anchor_before(language::Point::new(line, 10)),
                buffer.anchor_after(language::Point::new(line, 3))
                    ..buffer.anchor_before(language::Point::new(line, 3 + name.len() as u32)),
            )
        });
        Call {
            item: CallHierarchyItem {
                buffer: buffer.clone(),
                server_id: lsp::LanguageServerId(0),
                name: name.to_string(),
                kind: lsp::SymbolKind::FUNCTION,
                detail: None,
                range,
                selection_range: selection_range.clone(),
                data: None,
            },
            target: Location {
                buffer,
                range: selection_range,
            },
            site_count: 1,
            label: None,
            display: crate::CallDisplay::default(),
        }
    }
}
