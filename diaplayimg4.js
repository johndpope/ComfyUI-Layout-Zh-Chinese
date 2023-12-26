import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

const ART_GALLERY_NODE = "ArtGallery_Zho";

app.registerExtension({
    name: "zho.Combo++",
    init() {
        $el("style", {
            textContent: `
                .zho-combo-image {
                    display: block;
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

        const contextMenuHook = {};

        // After an element is created for an item, add an image if it has one
        contextMenuHook["addItem"] = function (el, menu, [name, value, options]) {
            if (el && value && typeof value === "object" && "image" in value) {
                el.textContent += " *";
                $el("div.zho-combo-image", {
                    parent: el,
                    style: {
                        backgroundImage: `url(/zho/view/${encodeURIComponent(value.image)})`,
                    },
                });
            }
        };

        app.ui.contextMenu.hooks.push(contextMenuHook);
    },
});
