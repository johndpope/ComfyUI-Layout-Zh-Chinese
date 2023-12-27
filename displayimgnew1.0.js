function ARTISTS_IMAGEUPLOAD(node, inputName, inputData, app) {
		const imageWidget = node.widgets.find((w) => w.name === (inputData[1]?.widget ?? "image"));
		let artistuploadWidget;

		function showImage(name) {
			const img = new Image();
			img.onload = () => {
				node.imgs = [img];
				app.graph.setDirtyCanvas(true);
			};
			let folder_separator = name.lastIndexOf("/");
			let subfolder = "";
			if (folder_separator > -1) {
				subfolder = name.substring(0, folder_separator);
				name = name.substring(folder_separator + 1);
			}
			img.src = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=artists${app.getPreviewFormatParam()}${app.getRandParam()}`);
			node.setSizeForImage?.();
		}

		var default_value = imageWidget.value;
		Object.defineProperty(imageWidget, "value", {
			set : function(value) {
				this._real_value = value;
			},

			get : function() {
				let value = "";
				if (this._real_value) {
					value = this._real_value;
				} else {
					return default_value;
				}

				if (value.filename) {
					let real_value = value;
					value = "";
					if (real_value.subfolder) {
						value = real_value.subfolder + "/";
					}

					value += real_value.filename;

					if(real_value.type && real_value.type !== "input")
						value += ` [${real_value.type}]`;
				}
				return value;
			}
		});

		// Add our own callback to the combo widget to render an image when it changes
		const cb = node.callback;
		imageWidget.callback = function () {
			showImage(imageWidget.value);
			if (cb) {
				return cb.apply(this, arguments);
			}
		};

		// On load if we have a value then render the image
		// The value isnt set immediately so we need to wait a moment
		// No change callbacks seem to be fired on initial setting of the value
		requestAnimationFrame(() => {
			if (imageWidget.value) {
				showImage(imageWidget.value);
			}
		});

		async function uploadFile(file, updateNode, pasted = false) {
			try {
				// Wrap file in formdata so it includes filename
				const body = new FormData();
				body.append("image", file);
				if (pasted) body.append("subfolder", "pasted");
				const resp = await api.fetchApi("/upload/image", {
					method: "POST",
					body,
				});

				if (resp.status === 200) {
					const data = await resp.json();
					// Add the file to the dropdown list and update the widget value
					let path = data.name;
					if (data.subfolder) path = data.subfolder + "/" + path;

					if (!imageWidget.options.values.includes(path)) {
						imageWidget.options.values.push(path);
					}

					if (updateNode) {
						showImage(path);
						imageWidget.value = path;
					}
				} else {
					alert(resp.status + " - " + resp.statusText);
				}
			} catch (error) {
				alert(error);
			}
		}

		const fileInput = document.createElement("input");
		Object.assign(fileInput, {
			type: "file",
			accept: "image/jpeg,image/png,image/webp",
			style: "display: none",
			onchange: async () => {
				if (fileInput.files.length) {
					await uploadFile(fileInput.files[0], true);
				}
			},
		});
		document.body.append(fileInput);

		// Create the button widget for selecting the files
		artistuploadWidget = node.addWidget("button", inputName, "image", () => {
			fileInput.click();
		});
		artistuploadWidget.label = "choose file to upload";
		artistuploadWidget.serialize = false;


		return { widget: artistuploadWidget };
	},
