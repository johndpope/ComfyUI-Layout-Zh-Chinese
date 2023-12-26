import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { $el } from "../../../scripts/ui.js";
import { api } from "../../../scripts/api.js";

const ARTIST_LOADER = "Artist_Zho";

function getType(node) {
	return "atszho";
}

app.registerExtension({
	name: "zho.Combo++",
	init() {
		$el("style", {
			textContent: `
				.litemenu-entry:hover .zho-combo-image {
					display: block;
				}
				.zho-combo-image {
					display: none;
					position: absolute;
					left: 0;
					top: 0;
					transform: translate(-100%, 0);
					width: 256px;
					height: 256px;
					background-size: cover;
					background-position: center;
					filter: brightness(65%);
				}
			`,
			parent: document.body,
		});


		// Ensure hook callbacks are available
		const getOrSet = (target, name, create) => {
			if (name in target) return target[name];
			return (target[name] = create());
		};
		const symbol = getOrSet(window, "__zho__", () => Symbol("__zho__"));
		const store = getOrSet(window, symbol, () => ({}));
		const contextMenuHook = getOrSet(store, "contextMenuHook", () => ({}));
		for (const e of ["ctor", "preAddItem", "addItem"]) {
			if (!contextMenuHook[e]) {
				contextMenuHook[e] = [];
			}
		}
		// // Checks if this is a custom combo item
		const isCustomItem = (value) => value && typeof value === "object" && "image" in value && value.content;
		// Simple check for what separator to split by
		const splitBy = (navigator.platform || navigator.userAgent).includes("Win") ? /\/|\\/ : /\//;


		// After an element is created for an item, add an image if it has one
		contextMenuHook["addItem"].push(function (el, menu, [name, value, options]) {
			if (el && isCustomItem(value) && value?.image && !value.submenu) {
				el.textContent += " *";
				$el("div.zho-combo-image", {
					parent: el,
					style: {
						backgroundImage: `url(/zho/view/${encodeURIComponent(value.image)})`,
					},
				});
			}
		});

		function buildMenu(widget, values) {
		}
		
		// Override COMBO widgets to patch their values
		const combo = ComfyWidgets["COMBO"];
		ComfyWidgets["COMBO"] = function (node, inputName, inputData) {
			const type = inputData[0];
			const res = combo.apply(this, arguments);
			if (isCustomItem(type[0])) {
				let value = res.widget.value;
				let values = res.widget.options.values;
				let menu = null;

				// Override the option values to check if we should render a menu structure
				Object.defineProperty(res.widget.options, "values", {
					get() {
						if (submenuSetting.value) {
							if (!menu) {
								// Only build the menu once
								menu = buildMenu(res.widget, values);
							}
							return menu;
						}
						return values;
					},
					set(v) {
						// Options are changing (refresh) so reset the menu so it can be rebuilt if required
						values = v;
						menu = null;
					},
				});

				Object.defineProperty(res.widget, "value", {
					get() {
						// HACK: litegraph supports rendering items with "content" in the menu, but not on the widget
						// This detects when its being called by the widget drawing and just returns the text
						// Also uses the content for the same image replacement value
						if (res.widget) {
							const stack = new Error().stack;
							if (stack.includes("drawNodeWidgets") || stack.includes("saveImageExtraOutput")) {
								return (value || type[0]).content;
							}
						}
						return value;
					},
					set(v) {
						if (v?.submenu) {
							// Dont allow selection of submenus
							return;
						}
						value = v;
					},
				});
			}

			return res;
		};
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const isATSL = nodeType.comfyClass === ARTIST_LOADER;
		if isATSL {
			const onAdded = nodeType.prototype.onAdded;
			nodeType.prototype.onAdded = function () {
				onAdded?.apply(this, arguments);
				const { widget: exampleList } = ComfyWidgets["COMBO"](this, "example", [[""]], app);

				let exampleWidget;

				const get = async (route, suffix) => {
					const url = encodeURIComponent(`${getType(nodeType)}${suffix || ""}`);
					return await api.fetchApi(`/zho/${route}/${url}`);
				};

				const getExample = async () => {
					if (exampleList.value === "[none]") {
						if (exampleWidget) {
							exampleWidget.inputEl.remove();
							exampleWidget = null;
							this.widgets.length -= 1;
						}
						return;
					}

					const v = this.widgets[0].value.content;
					const pos = v.lastIndexOf(".");
					const name = v.substr(0, pos);

					const example = await (await get("view", `/${name}/${exampleList.value}`)).text();
					if (!exampleWidget) {
						exampleWidget = ComfyWidgets["STRING"](this, "prompt", ["STRING", { multiline: true }], app).widget;
						exampleWidget.inputEl.readOnly = true;
						exampleWidget.inputEl.style.opacity = 0.6;
					}
					exampleWidget.value = example;
				};

				const exampleCb = exampleList.callback;
				exampleList.callback = function () {
					getExample();
					return exampleCb?.apply(this, arguments) ?? exampleList.value;
				};

				const listExamples = async () => {
					exampleList.disabled = true;
					exampleList.options.values = ["[none]"];
					exampleList.value = "[none]";
					let examples = [];
					if (this.widgets[0].value?.content) {
						try {
							examples = await (await get("examples", `/${this.widgets[0].value.content}`)).json();
						} catch (error) {}
					}
					exampleList.options.values = ["[none]", ...examples];
					exampleList.callback();
					exampleList.disabled = !examples.length;
					app.graph.setDirtyCanvas(true, true);
				};

				const modelWidget = this.widgets[0];
				const modelCb = modelWidget.callback;
				let prev = undefined;
				modelWidget.callback = function () {
					const ret = modelCb?.apply(this, arguments) ?? modelWidget.value;
					let v = ret;
					if (ret?.content) {
						v = ret.content;
					}
					if (prev !== v) {
						listExamples();
						prev = v;
					}
					return ret;
				};
				setTimeout(() => {
					modelWidget.callback();
				}, 30);
			};
		}

		
	},
});
