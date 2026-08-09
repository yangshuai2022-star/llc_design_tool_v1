# LLC GUI V7.1

This revision is a GUI information-architecture pass only. Numerical LLC/PFC/Vienna models are unchanged.

## Main-window changes

- Global LLC inputs are now a dockable panel instead of a permanent horizontal splitter.
- `F4`: global LLC design parameter dock.
- `F8`: run log dock (hidden by default).
- `F9`: focus mode.
- `Ctrl+R`: run the analysis associated with the current LLC page.
- Run log is no longer a permanent top-level tab.

## Digital-control page

- Compact two-row clickable control diagram.
- Context-sensitive parameter inspector: only one signal-chain stage is shown at a time.
- Inspector can be hidden independently to maximize Bode width.
- Clicking a block selects its parameters and its corresponding Bode view.
- Bode defaults to open-loop stability only.
- Detailed equations/results moved to a separate tab instead of permanently consuming plot height.
- Responsive Vout sensing schematic avoids clipping in narrow inspectors.

## V7.1.1 - Design parameter toggle UX
- The top toolbar “设计参数” action is now the single show/hide control for the global LLC parameter dock.
- First click hides the dock; the next click shows it again. F4 performs the same toggle.
- The action text follows the state: “隐藏设计参数” / “显示设计参数”.
- The dock close (X) button was removed to avoid a second, inconsistent close path.

## V7.1.2 - Signal-chain readability
- Enlarged the overview blocks, labels and arrows while keeping the diagram compact.
- Reworked orthogonal routing for the LLC forward and feedback signal paths.
- Increased vector text hierarchy and role-based block coloring.

## V7.1.3 - Zoom / focus / full-screen signal chain
- The LLC digital-control overview is now a true zoomable vector canvas.
- Mouse wheel zoom, hand panning, Fit, 100%, +/- controls and double-click-blank-to-fit are supported.
- Added a full-screen control-chain inspector with the same interactive selection linkage.
- Selecting a block highlights that block, its direct neighbours and the related signal path while dimming unrelated paths.
- Added a context-sensitive enlarged sub-diagram above the parameter page for Controller, FM LUT, PWM timing, LLC plant, analog sensing and ADC/averaging.
- Main diagram selection, local parameter page and Bode group remain linked.
