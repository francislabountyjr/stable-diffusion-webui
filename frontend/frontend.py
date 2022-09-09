import gradio as gr
from frontend.css_and_js import css, js, call_JS, js_parse_prompt, js_copy_txt2img_output
from frontend.job_manager import JobManager
import frontend.ui_functions as uifn
import uuid


def draw_gradio_ui(opt, img2img=lambda x: x, txt2img=lambda x: x, imgproc=lambda x: x, txt2img_defaults={},
                   RealESRGAN=True, GFPGAN=True, LDSR=True,
                   txt2img_toggles={}, txt2img_toggle_defaults='k_euler', show_embeddings=False, img2img_defaults={},
                   img2img_toggles={}, img2img_toggle_defaults={}, sample_img2img=None, img2img_mask_modes=None,
                   img2img_resize_modes=None, imgproc_defaults={},imgproc_mode_toggles={},user_defaults={}, run_GFPGAN=lambda x: x, run_RealESRGAN=lambda x: x,
                   txt_interp_toggles={}, txt_interp_toggle_defaults={}, disco_anim_toggles={}, disco_anim_toggle_defaults={},
                   txt_interp=lambda x: x, disco_anim=lambda x: x, txt_interp_defaults={}, disco_anim_defaults={}, stop_anim=lambda x: x,
                   job_manager: JobManager = None) -> gr.Blocks:
    with gr.Blocks(css=css(opt), analytics_enabled=False, title="Stable Diffusion WebUI") as demo:
        with gr.Tabs(elem_id='tabss') as tabs:
            with gr.TabItem("Text-to-Image", id='txt2img_tab'):
                with gr.Row(elem_id="prompt_row"):
                    txt2img_prompt = gr.Textbox(label="Prompt", 
                    elem_id='prompt_input',
                    placeholder="A corgi wearing a top hat as an oil painting.", 
                    lines=1,
                    max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25, 
                    value=txt2img_defaults['prompt'], 
                    show_label=False).style()
                    
                with gr.Row(elem_id='body').style(equal_height=False):
                    with gr.Column():

                        txt2img_height = gr.Slider(minimum=64, maximum=1024, step=64, label="Height", value=txt2img_defaults["height"])
                        txt2img_width = gr.Slider(minimum=64, maximum=1024, step=64, label="Width", value=txt2img_defaults["width"])
                        txt2img_cfg = gr.Slider(minimum=-30.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=txt2img_defaults['cfg_scale'])
                        txt2img_dynamic_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Dynamic Threshold', value=txt2img_defaults['dynamic_threshold'])
                        txt2img_static_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Static Threshold', value=txt2img_defaults['static_threshold'])
                        txt2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, max_lines=1, value=txt2img_defaults["seed"])                    
                        txt2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1, label='Batch count (how many batches of images to generate)', value=txt2img_defaults['n_iter'])
                        txt2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=txt2img_defaults['batch_size'])

                        txt2img_job_ui = job_manager.draw_gradio_ui() if job_manager else None

                        txt2img_dimensions_info_text_box = gr.Textbox(label="Aspect ratio (4:3 = 1.333 | 16:9 = 1.777 | 21:9 = 2.333)", interactive=False)
                    with gr.Column():
                        with gr.Box():
                            output_txt2img_gallery = gr.Gallery(label="Images", elem_id="txt2img_gallery_output").style(
                                grid=[4, 4])
                            gr.Markdown(
                                "Select an image from the gallery, then click one of the buttons below to perform an action.")
                            with gr.Row(elem_id='txt2img_actions_row'):
                                gr.Button("Copy to clipboard").click(fn=None,
                                                                     inputs=output_txt2img_gallery,
                                                                     outputs=[],
                                                                     # _js=js_copy_to_clipboard( 'txt2img_gallery_output')
                                                                     )
                                output_txt2img_copy_to_input_btn = gr.Button("Push to img2img")
                                output_txt2img_to_imglab = gr.Button("Send to Lab", visible=True)

                        output_txt2img_params = gr.Highlightedtext(label="Generation parameters", interactive=False,
                                                                   elem_id='highlight')
                        with gr.Group():
                            with gr.Row(elem_id='txt2img_output_row'):
                                output_txt2img_copy_params = gr.Button("Copy full parameters").click(
                                    inputs=[output_txt2img_params], outputs=[],
                                    _js=js_copy_txt2img_output,
                                    fn=None, show_progress=False)
                                output_txt2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                output_txt2img_copy_seed = gr.Button("Copy only seed").click(
                                    inputs=[output_txt2img_seed], outputs=[],
                                    _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                            output_txt2img_stats = gr.HTML(label='Stats')
                    with gr.Column():
                        txt2img_btn = gr.Button("Generate", full_width=True, elem_id="generate", variant="primary")
                        txt2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=txt2img_defaults['ddim_steps'])
                        txt2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=txt2img_defaults['sampler_name'])

                        with gr.Tabs():
                            with gr.TabItem('Simple'):
                                txt2img_submit_on_enter = gr.Radio(['Yes', 'No'], label="Submit on enter? (no means multiline)", value=txt2img_defaults['submit_on_enter'], interactive=True)
                                txt2img_submit_on_enter.change(lambda x: gr.update(max_lines=1 if x == 'Single' else 25) , txt2img_submit_on_enter, txt2img_prompt)
                            with gr.TabItem('Advanced'):
                                txt2img_toggles = gr.CheckboxGroup(label='', choices=txt2img_toggles, value=txt2img_toggle_defaults, type="index")
                                txt2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x4plus', visible=RealESRGAN is not None) # TODO: Feels like I shouldnt slot it in here.
                                txt2img_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=txt2img_defaults['ddim_eta'], visible=False)
                                txt2img_variant_amount = gr.Slider(minimum=0.0, maximum=1.0, label='Variation Amount',
                                                                   value=txt2img_defaults['variant_amount'])
                                txt2img_variant_seed = gr.Textbox(label="Variant Seed (blank to randomize)", lines=1,
                                                                  max_lines=1, value=txt2img_defaults["variant_seed"])
                        txt2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)

                txt2img_func = txt2img
                txt2img_inputs = [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles, txt2img_realesrgan_model_name, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfg, txt2img_dynamic_threshold, txt2img_static_threshold, txt2img_seed, txt2img_variant_amount, txt2img_variant_seed, txt2img_height, txt2img_width, txt2img_embeddings]
                txt2img_outputs = [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]

                # If a JobManager was passed in then wrap the Generate functions
                if txt2img_job_ui:
                    txt2img_func, txt2img_inputs, txt2img_outputs = txt2img_job_ui.wrap_func(
                        func=txt2img_func,
                        inputs=txt2img_inputs,
                        outputs=txt2img_outputs
                    )

                txt2img_btn.click(
                    txt2img_func,
                    txt2img_inputs,
                    txt2img_outputs
                )
                txt2img_prompt.submit(
                    txt2img_func,
                    txt2img_inputs,
                    txt2img_outputs
                )

                # txt2img_width.change(fn=uifn.update_dimensions_info, inputs=[txt2img_width, txt2img_height], outputs=txt2img_dimensions_info_text_box)
                # txt2img_height.change(fn=uifn.update_dimensions_info, inputs=[txt2img_width, txt2img_height], outputs=txt2img_dimensions_info_text_box)

                live_prompt_params = [txt2img_prompt, txt2img_width, txt2img_height, txt2img_steps, txt2img_seed,
                                      txt2img_batch_count, txt2img_cfg]
                txt2img_prompt.change(
                    fn=None,
                    inputs=live_prompt_params,
                    outputs=live_prompt_params,
                    _js=js_parse_prompt
                )

            with gr.TabItem("Image-to-Image", id="img2img_tab"):
                with gr.Row(elem_id="prompt_row"):
                    img2img_prompt = gr.Textbox(label="Prompt",
                                                elem_id='img2img_prompt_input',
                                                placeholder="A fantasy landscape, trending on artstation.",
                                                lines=1,
                                                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25,
                                                value=img2img_defaults['prompt'],
                                                show_label=False).style()

                    img2img_btn_mask = gr.Button("Generate", variant="primary", visible=False,
                                                 elem_id="img2img_mask_btn")
                    img2img_btn_editor = gr.Button("Generate", variant="primary", elem_id="img2img_edit_btn")
                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        gr.Markdown('#### Img2Img Input')
                        img2img_image_mask = gr.Image(
                            value=sample_img2img,
                            source="upload",
                            interactive=True,
                            type="pil", tool="sketch",
                            elem_id="img2img_mask",
                            image_mode="RGBA"
                        )
                        img2img_image_editor = gr.Image(
                            value=sample_img2img,
                            source="upload",
                            interactive=True,
                            type="pil",
                            tool="select",
                            visible=False,
                            image_mode="RGBA",
                            elem_id="img2img_editor"
                        )

                        with gr.Tabs():
                            with gr.TabItem("Editor Options"):
                                with gr.Row():
                                    # disable Uncrop for now
                                    # choices=["Mask", "Crop", "Uncrop"]
                                    img2img_image_editor_mode = gr.Radio(choices=["Mask", "Crop"],
                                                                         label="Image Editor Mode",
                                                                         value="Mask", elem_id='edit_mode_select',
                                                                         visible=True)
                                    img2img_mask = gr.Radio(choices=["Keep masked area", "Regenerate only masked area"],
                                                            label="Mask Mode", type="index",
                                                            value=img2img_mask_modes[img2img_defaults['mask_mode']],
                                                            visible=True)

                                    img2img_mask_blur_strength = gr.Slider(minimum=1, maximum=10, step=1,
                                                                           label="How much blurry should the mask be? (to avoid hard edges)",
                                                                           value=3, visible=False)

                                    img2img_resize = gr.Radio(label="Resize mode",
                                                              choices=["Just resize", "Crop and resize",
                                                                       "Resize and fill"],
                                                              type="index",
                                                              value=img2img_resize_modes[
                                                                  img2img_defaults['resize_mode']], visible=False)

                                img2img_painterro_btn = gr.Button("Advanced Editor")
                            with gr.TabItem("Hints"):
                                img2img_help = gr.Markdown(visible=False, value=uifn.help_text)

                    with gr.Column():
                        gr.Markdown('#### Img2Img Results')
                        output_img2img_gallery = gr.Gallery(label="Images", elem_id="img2img_gallery_output").style(
                            grid=[4, 4, 4])
                        img2img_job_ui = job_manager.draw_gradio_ui() if job_manager else None
                        with gr.Tabs():
                            with gr.TabItem("Generated image actions", id="img2img_actions_tab"):
                                gr.Markdown("Select an image, then press one of the buttons below")
                                with gr.Row():
                                    output_img2img_copy_to_clipboard_btn = gr.Button("Copy to clipboard")
                                    output_img2img_copy_to_input_btn = gr.Button("Push to img2img input")
                                    output_img2img_copy_to_mask_btn = gr.Button("Push to img2img input mask")

                                gr.Markdown("Warning: This will clear your current image and mask settings!")
                            with gr.TabItem("Output info", id="img2img_output_info_tab"):
                                output_img2img_params = gr.Textbox(label="Generation parameters")
                                with gr.Row():
                                    output_img2img_copy_params = gr.Button("Copy full parameters").click(
                                        inputs=output_img2img_params, outputs=[],
                                        _js='(x) => {navigator.clipboard.writeText(x.replace(": ",":"))}', fn=None,
                                        show_progress=False)
                                    output_img2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                    output_img2img_copy_seed = gr.Button("Copy only seed").click(
                                        inputs=output_img2img_seed, outputs=[],
                                        _js=call_JS("gradioInputToClipboard"), fn=None, show_progress=False)
                                output_img2img_stats = gr.HTML(label='Stats')

                gr.Markdown('# img2img settings')

                with gr.Row():
                    with gr.Column():
                        img2img_width = gr.Slider(minimum=64, maximum=1024, step=64, label="Width",
                                                  value=img2img_defaults["width"])
                        img2img_height = gr.Slider(minimum=64, maximum=1024, step=64, label="Height",
                                                   value=img2img_defaults["height"])
                        img2img_cfg = gr.Slider(minimum=-30.0, maximum=30.0, step=0.5,
                                                label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)',
                                                value=img2img_defaults['cfg_scale'], elem_id='cfg_slider')
                        img2img_dynamic_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Dynamic Threshold', value=img2img_defaults['dynamic_threshold'])
                        img2img_static_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Static Threshold', value=img2img_defaults['static_threshold'])
                        img2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, max_lines=1,
                                                  value=img2img_defaults["seed"])
                        img2img_batch_count = gr.Slider(minimum=1, maximum=50, step=1,
                                                        label='Batch count (how many batches of images to generate)',
                                                        value=img2img_defaults['n_iter'])

                        img2img_dimensions_info_text_box = gr.Textbox(label="Aspect ratio (4:3 = 1.333 | 16:9 = 1.777 | 21:9 = 2.333)", interactive=False)
                    with gr.Column():
                        img2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps",
                                                  value=img2img_defaults['ddim_steps'])

                        img2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)',
                                                       choices=["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler',
                                                                'k_heun', 'k_lms'],
                                                       value=img2img_defaults['sampler_name'])

                        img2img_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength',
                                                      value=img2img_defaults['denoising_strength'])

                        img2img_toggles = gr.CheckboxGroup(label='', choices=img2img_toggles,
                                                           value=img2img_toggle_defaults, type="index")

                        img2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model',
                                                                    choices=['RealESRGAN_x4plus',
                                                                             'RealESRGAN_x4plus_anime_6B'],
                                                                    value='RealESRGAN_x4plus',
                                                                    visible=RealESRGAN is not None)  # TODO: Feels like I shouldnt slot it in here.

                        img2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)

                img2img_image_editor_mode.change(
                    uifn.change_image_editor_mode,
                    [img2img_image_editor_mode,
                     img2img_image_editor,
                     img2img_image_mask,
                     img2img_resize,
                     img2img_width,
                     img2img_height
                     ],
                    [img2img_image_editor, img2img_image_mask, img2img_btn_editor, img2img_btn_mask,
                     img2img_painterro_btn, img2img_mask, img2img_mask_blur_strength]
                )

                # img2img_image_editor_mode.change(
                #     uifn.update_image_mask,
                #     [img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                #     img2img_image_mask
                # )

                output_txt2img_copy_to_input_btn.click(
                    uifn.copy_img_to_input,
                    [output_txt2img_gallery],
                    [img2img_image_editor, img2img_image_mask, tabs],
                    _js=call_JS("moveImageFromGallery",
                                fromId="txt2img_gallery_output",
                                toId="img2img_editor")
                )

                output_img2img_copy_to_input_btn.click(
                    uifn.copy_img_to_edit,
                    [output_img2img_gallery],
                    [img2img_image_editor, tabs, img2img_image_editor_mode],
                    _js=call_JS("moveImageFromGallery",
                                fromId="img2img_gallery_output",
                                toId="img2img_editor")
                )
                output_img2img_copy_to_mask_btn.click(
                    uifn.copy_img_to_mask,
                    [output_img2img_gallery],
                    [img2img_image_mask, tabs, img2img_image_editor_mode],
                    _js=call_JS("moveImageFromGallery",
                                fromId="img2img_gallery_output",
                                toId="img2img_editor")
                )

                output_img2img_copy_to_clipboard_btn.click(fn=None, inputs=output_img2img_gallery, outputs=[],
                                                           _js=call_JS("copyImageFromGalleryToClipboard",
                                                                       fromId="img2img_gallery_output")
                                                           )

                img2img_func = img2img
                img2img_inputs = [img2img_prompt, img2img_image_editor_mode, img2img_mask,
                                  img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles,
                                  img2img_realesrgan_model_name, img2img_batch_count, img2img_cfg,
                                  img2img_denoising, img2img_dynamic_threshold, img2img_static_threshold,
                                  img2img_seed, img2img_height, img2img_width, img2img_resize,
                                  img2img_embeddings]
                img2img_outputs = [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats]

                # If a JobManager was passed in then wrap the Generate functions
                if img2img_job_ui:
                    img2img_func, img2img_inputs, img2img_outputs = img2img_job_ui.wrap_func(
                        func=img2img_func,
                        inputs=img2img_inputs,
                        outputs=img2img_outputs,
                    )

                img2img_btn_mask.click(
                    img2img_func,
                    img2img_inputs,
                    img2img_outputs
                )

                def img2img_submit_params():
                    # print([img2img_prompt, img2img_image_editor_mode, img2img_mask,
                    #              img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles,
                    #              img2img_realesrgan_model_name, img2img_batch_count, img2img_cfg,
                    #              img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize,
                    #              img2img_image_editor, img2img_image_mask, img2img_embeddings])
                    return (img2img_func,
                            img2img_inputs,
                            img2img_outputs)

                img2img_btn_editor.click(*img2img_submit_params())

                # GENERATE ON ENTER
                img2img_prompt.submit(None, None, None,
                                      _js=call_JS("clickFirstVisibleButton",
                                                  rowId="prompt_row"))

                img2img_painterro_btn.click(None,
                                            [img2img_image_editor, img2img_image_mask, img2img_image_editor_mode],
                                            [img2img_image_editor, img2img_image_mask],
                                            _js=call_JS("Painterro.init", toId="img2img_editor")
                                            )

                img2img_width.change(fn=uifn.update_dimensions_info, inputs=[img2img_width, img2img_height],
                                     outputs=img2img_dimensions_info_text_box)
                img2img_height.change(fn=uifn.update_dimensions_info, inputs=[img2img_width, img2img_height],
                                      outputs=img2img_dimensions_info_text_box)

            with gr.TabItem("Text Interpolation", id='txt_interp_tab'):
                with gr.Row(elem_id="prompt_row"):
                    txt_interp_prompt = gr.Textbox(label="Prompt", 
                    elem_id='prompt_input',
                    placeholder="An epic matte painting of a wizards potion room, featured on artstation\nAn epic matte painting of a dragons lair, featured on artstation", 
                    lines=1,
                    max_lines=100,
                    value=txt_interp_defaults['prompt'], 
                    show_label=False).style()
                    
                with gr.Row(elem_id='body').style(equal_height=False):
                    with gr.Column():
                        txt_interp_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=txt_interp_defaults["height"])
                        txt_interp_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=txt_interp_defaults["width"])
                        txt_interp_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=txt_interp_defaults['cfg_scale'])
                        txt_interp_dynamic_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Dynamic Threshold', value=txt_interp_defaults['dynamic_threshold'])
                        txt_interp_static_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Static Threshold', value=txt_interp_defaults['static_threshold'])
                        txt_interp_degrees_per_second = gr.Slider(minimum=1, maximum=360, step=1, label='Degrees Per Second', value=txt_interp_defaults['degrees_per_second'])
                        txt_interp_frames_per_second = gr.Slider(minimum=1, maximum=360, step=1, label='Frames Per Second', value=txt_interp_defaults['frames_per_second'])
                        txt_interp_project_name = gr.Textbox(label="Project Name", lines=1, max_lines=1, value=txt_interp_defaults["project_name"])
                        txt_interp_seeds = gr.Textbox(label="Seeds (blank or None to randomize, seperate with newline)", lines=1, max_lines=100, value=txt_interp_defaults["seeds"])
                        txt_interp_batch_size = gr.Slider(minimum=1, maximum=20, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=txt_interp_defaults['batch_size'])
                    with gr.Column():
                        output_txt_interp_progress_images = gr.Image()
                        txt_interp_job_ui = job_manager.draw_gradio_ui()
                        # with gr.Row():
                        #     with gr.Group():
                        #         output_txt_interp_progress = gr.Textbox(label='Progress Status', interactive=False)
                    with gr.Column():
                        txt_interp_btn = gr.Button("Generate", full_width=True, elem_id="generate", variant="primary")
                        txt_interp_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=txt_interp_defaults['ddim_steps'])
                        txt_interp_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=txt_interp_defaults['sampler_name'])
                        with gr.Tabs():
                            with gr.TabItem('Options'):
                                txt_interp_toggles = gr.CheckboxGroup(label='', choices=txt_interp_toggles, value=txt_interp_toggle_defaults, type="index")
                                txt_interp_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x4plus', visible=RealESRGAN is not None) # TODO: Feels like I shouldnt slot it in here.
                                txt_interp_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=txt_interp_defaults['ddim_eta'], visible=False)
                        txt_interp_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)

                txt_interp_func = txt_interp
                txt_interp_inputs = [txt_interp_prompt, txt_interp_steps, txt_interp_sampling, txt_interp_toggles, txt_interp_realesrgan_model_name, txt_interp_ddim_eta, txt_interp_batch_size, txt_interp_cfg, txt_interp_dynamic_threshold, txt_interp_static_threshold, txt_interp_degrees_per_second, txt_interp_frames_per_second, txt_interp_project_name, txt_interp_seeds, txt_interp_height, txt_interp_width, txt_interp_embeddings]
                txt_interp_outputs = output_txt_interp_progress_images

                # txt_interp_func, txt_interp_inputs, txt_interp_outputs = txt_interp_job_ui.wrap_func(
                #         func=txt_interp_func,
                #         inputs=txt_interp_inputs,
                #         outputs=txt_interp_outputs,
                #     )

                txt_interp_btn.click(
                    txt_interp_func,
                    txt_interp_inputs,
                    txt_interp_outputs
                )
                txt_interp_prompt.submit(
                    txt_interp_func,
                    txt_interp_inputs,
                    txt_interp_outputs
                )

            with gr.TabItem("Disco Animation", id='disco_anim_tab'):
                with gr.Row(elem_id="prompt_row"):
                    disco_anim_prompt = gr.Textbox(label="Prompt", 
                    elem_id='prompt_input',
                    placeholder="An epic matte painting of a wizards potion room, featured on artstation\nAn epic matte painting of a dragons lair, featured on artstation", 
                    lines=1,
                    max_lines=100, 
                    value=disco_anim_defaults['prompt'], 
                    show_label=False).style()
                    
                with gr.Row(elem_id='body').style(equal_height=False):
                    with gr.Column():
                        disco_anim_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=disco_anim_defaults["height"])
                        disco_anim_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=disco_anim_defaults["width"])
                        disco_anim_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=disco_anim_defaults['cfg_scale'])
                        disco_anim_dynamic_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Dynamic Threshold', value=disco_anim_defaults['dynamic_threshold'])
                        disco_anim_static_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, label='Static Threshold', value=disco_anim_defaults['static_threshold'])
                        disco_anim_prev_frame_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Previous Frame Denoising Strength', value=disco_anim_defaults['prev_frame_denoising_strength'])
                        disco_anim_noise_between_frames = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label='Amount of noise to inject in between frames', value=disco_anim_defaults['noise_between_frames'])
                        disco_anim_degrees_per_second = gr.Slider(minimum=1, maximum=360, step=1, label='Degrees Per Second (if interpolating between prompts)', value=disco_anim_defaults['degrees_per_second'])
                        disco_anim_frames_per_second = gr.Slider(minimum=1, maximum=360, step=1, label='Frames Per Second  (if interpolating between prompts)', value=disco_anim_defaults['frames_per_second'])
                        disco_anim_project_name = gr.Textbox(label="Project Name", lines=1, max_lines=1, value=disco_anim_defaults["project_name"])
                        disco_anim_start_frame = gr.Number(precision=None, label="Start Frame (will use 0 if not resuming an animation)", value=0)
                        disco_anim_max_frames = gr.Number(precision=None, label="Max Frames in Animation", value=disco_anim_defaults['max_frames'])
                        disco_anim_seed = gr.Textbox(label="Seed (blank or None to randomize)", lines=1, max_lines=1, value='')
                        disco_anim_animation_mode = gr.Dropdown(label='Animation Mode', choices=["3D", "2D"], value=disco_anim_defaults['animation_mode']) # video mode WIP
                        disco_anim_interp_spline = gr.Dropdown(label='Spline Interpolation (Linear Recommended)', choices=["Linear", "Quadratic", "Cubic"], value=disco_anim_defaults['interp_spline'])
                        disco_anim_resize_mode = gr.Radio(label="Resize mode", choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value=img2img_resize_modes[disco_anim_defaults['resize_mode']])
                        disco_anim_color_match_mode = gr.Dropdown(label='Color Match Mode (if enabled)', choices=["RGB", "HSV", "LAB", "cycle"], value=disco_anim_defaults['color_match_mode'])
                        disco_anim_mix_factor = gr.Textbox(label="Amount of previous frame's latent to inject in between timesteps (1.0 - mix factor = amount mixed in)", lines=1, max_lines=1, value=disco_anim_defaults['mix_factor'])
                        with gr.Group():
                            disco_anim_angle = gr.Textbox(label='Angle', lines=1, max_lines=1, value=disco_anim_defaults['angle'])
                            disco_anim_zoom = gr.Textbox(label='Zoom', lines=1, max_lines=1, value=disco_anim_defaults['zoom'])
                            disco_anim_translation_x = gr.Textbox(label='Translation x', lines=1, max_lines=1, value=disco_anim_defaults['translation_x'])
                            disco_anim_translation_y = gr.Textbox(label='Translation y', lines=1, max_lines=1, value=disco_anim_defaults['translation_y'])
                            disco_anim_translation_z = gr.Textbox(label='Translation z', lines=1, max_lines=1, value=disco_anim_defaults['translation_z'])
                            disco_anim_rotation_3d_x = gr.Textbox(label='Rotation 3D x', lines=1, max_lines=1, value=disco_anim_defaults['rotation_3d_x'])
                            disco_anim_rotation_3d_y = gr.Textbox(label='Rotation 3D y', lines=1, max_lines=1, value=disco_anim_defaults['rotation_3d_y'])
                            disco_anim_rotation_3d_z = gr.Textbox(label='Rotation 3D z', lines=1, max_lines=1, value=disco_anim_defaults['rotation_3d_z'])
                            disco_anim_midas_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Angle', value=disco_anim_defaults['midas_weight'])
                            disco_anim_near_plane = gr.Slider(minimum=1, maximum=1000, step=1, label='Near Plane', value=disco_anim_defaults['near_plane'])
                            disco_anim_far_plane = gr.Slider(minimum=1, maximum=50000, step=1, label='Far Plane', value=disco_anim_defaults['far_plane'])
                            disco_anim_fov = gr.Slider(minimum=1, maximum=360, step=1, label='Field of View', value=disco_anim_defaults['fov'])
                            disco_anim_padding_mode= gr.Dropdown(label='Padding Mode', choices=["border"], value=disco_anim_defaults['padding_mode'])
                            disco_anim_sampling_mode = gr.Dropdown(label='Sampling Mode', choices=["bicubic"], value=disco_anim_defaults['sampling_mode'])
                            disco_anim_turbo_steps = gr.Slider(minimum=1, maximum=5, step=1, label='Turbo Steps', value=disco_anim_defaults['turbo_steps'])
                            disco_anim_turbo_preroll = gr.Slider(minimum=1, maximum=15, step=1, label='Turbo Preroll', value=disco_anim_defaults['turbo_preroll'])
                            disco_anim_vr_eye_angle = gr.Slider(minimum=1.0, maximum=10.0, step=0.01, label='vr Eye Angle', value=disco_anim_defaults['vr_eye_angle'])
                            disco_anim_vr_ipd = gr.Slider(minimum=1.0, maximum=10.0, step=0.01, label='vr IPD', value=disco_anim_defaults['vr_ipd'])
                        disco_anim_init_info = gr.Image(value=None, source="upload", interactive=True, type="pil", tool="select")
                        # with gr.Group():
                        #     disco_anim_extract_nth_frame = gr.Slider(minimum=1, maximum=100, step=1, label='Extract nth Frame', value=disco_anim_defaults['extract_nth_frame'])
                        #     disco_anim_video_init_flow_blend = gr.Slider(minimum=0, maximum=1, step=.01, label='Video Init Flow Blend', value=disco_anim_defaults['video_init_flow_blend'])
                        #     disco_anim_video_init_blend_mode= gr.Dropdown(label='Video Init Blend Mode', choices=['None', 'linear', 'optical flow'], value=disco_anim_defaults['video_init_blend_mode'])
                    with gr.Column():
                        output_disco_anim_progress_images = gr.Image()
                        with gr.Row():
                            with gr.Group():
                                output_disco_anim_progress = gr.Textbox(label='Progress Status', interactive=False)
                    with gr.Column():
                        disco_anim_btn = gr.Button("Generate", full_width=True, elem_id="generate", variant="primary")
                        # disco_anim_stop_anim = gr.Button("Stop Animation", full_width=True, elem_id="stop_animation", variant='primary')
                        disco_anim_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=disco_anim_defaults['ddim_steps'])
                        disco_anim_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value=disco_anim_defaults['sampler_name'])
                        with gr.Tabs():
                            with gr.Group():
                                disco_anim_toggles = gr.CheckboxGroup(label='', choices=disco_anim_toggles, value=disco_anim_toggle_defaults, type="index")
                                disco_anim_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=disco_anim_defaults['ddim_eta'], visible=False)
                        disco_anim_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)

                disco_anim_btn.click(
                    disco_anim,
                    [disco_anim_prompt, disco_anim_init_info, disco_anim_project_name, disco_anim_steps, disco_anim_sampling, disco_anim_toggles, disco_anim_ddim_eta, disco_anim_cfg, disco_anim_color_match_mode, disco_anim_dynamic_threshold, disco_anim_static_threshold, disco_anim_degrees_per_second, disco_anim_frames_per_second, disco_anim_prev_frame_denoising, disco_anim_noise_between_frames, disco_anim_mix_factor, disco_anim_start_frame, disco_anim_max_frames, disco_anim_animation_mode, disco_anim_interp_spline, disco_anim_angle, disco_anim_zoom, disco_anim_translation_x, disco_anim_translation_y, disco_anim_translation_z, disco_anim_rotation_3d_x, disco_anim_rotation_3d_y, disco_anim_rotation_3d_z, disco_anim_midas_weight, disco_anim_near_plane, disco_anim_far_plane, disco_anim_fov, disco_anim_padding_mode, disco_anim_sampling_mode, disco_anim_turbo_steps, disco_anim_turbo_preroll, disco_anim_vr_eye_angle, disco_anim_vr_ipd, disco_anim_seed, disco_anim_height, disco_anim_width, disco_anim_resize_mode, disco_anim_embeddings],
                    [output_disco_anim_progress_images,  output_disco_anim_progress] #, output_disco_anim_seed, output_disco_anim_params, output_disco_anim_stats]
                )
                disco_anim_prompt.submit(
                    disco_anim,
                    [disco_anim_prompt, disco_anim_init_info, disco_anim_project_name, disco_anim_steps, disco_anim_sampling, disco_anim_toggles, disco_anim_ddim_eta, disco_anim_cfg, disco_anim_color_match_mode, disco_anim_dynamic_threshold, disco_anim_static_threshold, disco_anim_degrees_per_second, disco_anim_frames_per_second, disco_anim_prev_frame_denoising, disco_anim_noise_between_frames, disco_anim_mix_factor, disco_anim_start_frame, disco_anim_max_frames, disco_anim_animation_mode, disco_anim_interp_spline, disco_anim_angle, disco_anim_zoom, disco_anim_translation_x, disco_anim_translation_y, disco_anim_translation_z, disco_anim_rotation_3d_x, disco_anim_rotation_3d_y, disco_anim_rotation_3d_z, disco_anim_midas_weight, disco_anim_near_plane, disco_anim_far_plane, disco_anim_fov, disco_anim_padding_mode, disco_anim_sampling_mode, disco_anim_turbo_steps, disco_anim_turbo_preroll, disco_anim_vr_eye_angle, disco_anim_vr_ipd, disco_anim_seed, disco_anim_height, disco_anim_width, disco_anim_resize_mode, disco_anim_embeddings],
                    [output_disco_anim_progress_images,  output_disco_anim_progress] #, output_disco_anim_seed, output_disco_anim_params, output_disco_anim_stats]
                )

                # disco_anim_stop_anim.click(
                #     stop_anim,
                #     [],
                #     []
                # )
            
            with gr.TabItem("Image Lab", id='imgproc_tab'):
                    gr.Markdown("Post-process results")
                    with gr.Row():
                        with gr.Column():
                            with gr.Tabs():
                                with gr.TabItem('Single Image'):
                                    imgproc_source = gr.Image(label="Source", source="upload", interactive=True, type="pil",elem_id="imglab_input")

            with gr.TabItem("Image Lab", id='imgproc_tab'):
                gr.Markdown("Post-process results")
                with gr.Row():
                    with gr.Column():
                        with gr.Tabs():
                            with gr.TabItem('Single Image'):
                                imgproc_source = gr.Image(label="Source", source="upload", interactive=True, type="pil",
                                                          elem_id="imglab_input")

                            # gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Effect strength",
                            #                            value=gfpgan_defaults['strength'])
                            # select folder with images to process
                            with gr.TabItem('Batch Process'):
                                imgproc_folder = gr.File(label="Batch Process", file_count="multiple", source="upload",
                                                         interactive=True, type="file")
                        imgproc_pngnfo = gr.Textbox(label="PNG Metadata", placeholder="PngNfo", visible=False,
                                                    max_lines=5)
                        with gr.Row():
                            imgproc_btn = gr.Button("Process", variant="primary")
                        gr.HTML("""
        <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
            <p><b>Upscale Modes Guide</b></p>
            <p></p>
            <p><b>RealESRGAN</b></p>
            <p>A 4X/2X fast upscaler that works well for stylized content, will smooth more detailed compositions.</p>
            <p><b>GoBIG</b></p>
            <p>A 2X upscaler that uses RealESRGAN to upscale the image and then slice it into small parts, each part gets diffused further by SD to create more details, great for adding and increasing details but will change the composition, might also fix issues like eyes etc, use the settings like img2img etc</p>
            <p><b>Latent Diffusion Super Resolution</b></p>
            <p>A 4X upscaler with high VRAM usage that uses a Latent Diffusion model to upscale the image, this will accentuate the details but won't change the composition, might introduce sharpening, great for textures or compositions with plenty of details, is slower.</p>
            <p><b>GoLatent</b></p>
            <p>A 8X upscaler with high VRAM usage, uses GoBig to add details and then uses a Latent Diffusion model to upscale the image, this will result in less artifacting/sharpeninng, use the settings to feed GoBig settings that will contribute to the result, this mode is considerbly slower</p>
        </div>
        """)
                    with gr.Column():
                        with gr.Tabs():
                            with gr.TabItem('Output'):
                                imgproc_output = gr.Gallery(label="Output", elem_id="imgproc_gallery_output")
                        with gr.Row(elem_id="proc_options_row"):
                            with gr.Box():
                                with gr.Column():
                                    gr.Markdown("<b>Processor Selection</b>")
                                    imgproc_toggles = gr.CheckboxGroup(label='', choices=imgproc_mode_toggles,
                                                                       type="index")
                                    # .change toggles to show options
                                    # imgproc_toggles.change()
                        with gr.Box(visible=False) as gfpgan_group:

                            gfpgan_defaults = {
                                'strength': 100,
                            }

                            if 'gfpgan' in user_defaults:
                                gfpgan_defaults.update(user_defaults['gfpgan'])
                            if GFPGAN is None:
                                gr.HTML("""
    <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
        <p><b> Please download GFPGAN to activate face fixing features</b>, instructions are available at the <a href='https://github.com/hlky/stable-diffusion-webui'>Github</a></p>
    </div>
    """)
                                # gr.Markdown("")
                                # gr.Markdown("<b> Please download GFPGAN to activate face fixing features</b>, instructions are available at the <a href='https://github.com/hlky/stable-diffusion-webui'>Github</a>")
                            with gr.Column():
                                gr.Markdown("<b>GFPGAN Settings</b>")
                                imgproc_gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001,
                                                                    label="Effect strength",
                                                                    value=gfpgan_defaults['strength'],
                                                                    visible=GFPGAN is not None)
                        with gr.Box(visible=False) as upscale_group:

                            if LDSR:
                                upscaleModes = ['RealESRGAN', 'GoBig', 'Latent Diffusion SR', 'GoLatent ']
                            else:
                                gr.HTML("""
    <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
        <p><b> Please download LDSR to activate more upscale features</b>, instructions are available at the <a href='https://github.com/hlky/stable-diffusion-webui'>Github</a></p>
    </div>
    """)
                                upscaleModes = ['RealESRGAN', 'GoBig']
                            with gr.Column():
                                gr.Markdown("<b>Upscaler Selection</b>")
                                imgproc_upscale_toggles = gr.Radio(label='', choices=upscaleModes, type="index",
                                                                   visible=RealESRGAN is not None, value='RealESRGAN')
                        with gr.Box(visible=False) as upscalerSettings_group:

                            with gr.Box(visible=True) as realesrgan_group:
                                with gr.Column():
                                    gr.Markdown("<b>RealESRGAN Settings</b>")
                                    imgproc_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model',
                                                                                interactive=RealESRGAN is not None,
                                                                                choices=['RealESRGAN_x4plus',
                                                                                         'RealESRGAN_x4plus_anime_6B',
                                                                                         'RealESRGAN_x2plus',
                                                                                         'RealESRGAN_x2plus_anime_6B'],
                                                                                value='RealESRGAN_x4plus',
                                                                                visible=RealESRGAN is not None)  # TODO: Feels like I shouldnt slot it in here.
                            with gr.Box(visible=False) as ldsr_group:
                                with gr.Row(elem_id="ldsr_settings_row"):
                                    with gr.Column():
                                        gr.Markdown("<b>Latent Diffusion Super Sampling Settings</b>")
                                        imgproc_ldsr_steps = gr.Slider(minimum=0, maximum=500, step=10,
                                                                       label="LDSR Sampling Steps",
                                                                       value=100, visible=LDSR is not None)
                                        imgproc_ldsr_pre_downSample = gr.Dropdown(
                                            label='LDSR Pre Downsample mode (Lower resolution before processing for speed)',
                                            choices=["None", '1/2', '1/4'], value="None", visible=LDSR is not None)
                                        imgproc_ldsr_post_downSample = gr.Dropdown(
                                            label='LDSR Post Downsample mode (aka SuperSampling)',
                                            choices=["None", "Original Size", '1/2', '1/4'], value="None",
                                            visible=LDSR is not None)
                            with gr.Box(visible=False) as gobig_group:
                                with gr.Row(elem_id="proc_prompt_row"):
                                    with gr.Column():
                                        gr.Markdown("<b>GoBig Settings</b>")
                                        imgproc_prompt = gr.Textbox(label="",
                                                                    elem_id='prompt_input',
                                                                    placeholder="A corgi wearing a top hat as an oil painting.",
                                                                    lines=1,
                                                                    max_lines=1,
                                                                    value=imgproc_defaults['prompt'],
                                                                    show_label=True,
                                                                    visible=RealESRGAN is not None)
                                        imgproc_sampling = gr.Dropdown(
                                            label='Sampling method (k_lms is default k-diffusion sampler)',
                                            choices=["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler',
                                                     'k_heun', 'k_lms'],
                                            value=imgproc_defaults['sampler_name'], visible=RealESRGAN is not None)
                                        imgproc_steps = gr.Slider(minimum=1, maximum=250, step=1,
                                                                  label="Sampling Steps",
                                                                  value=imgproc_defaults['ddim_steps'],
                                                                  visible=RealESRGAN is not None)
                                        imgproc_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5,
                                                                label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)',
                                                                value=imgproc_defaults['cfg_scale'],
                                                                visible=RealESRGAN is not None)
                                        imgproc_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                                                      label='Denoising Strength',
                                                                      value=imgproc_defaults['denoising_strength'],
                                                                      visible=RealESRGAN is not None)
                                        imgproc_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height",
                                                                   value=imgproc_defaults["height"],
                                                                   visible=False)  # not currently implemented
                                        imgproc_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width",
                                                                  value=imgproc_defaults["width"],
                                                                  visible=False)  # not currently implemented
                                        imgproc_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1,
                                                                  max_lines=1,
                                                                  value=imgproc_defaults["seed"],
                                                                  visible=RealESRGAN is not None)
                                        imgproc_btn.click(
                                            imgproc,
                                            [imgproc_source, imgproc_folder, imgproc_prompt, imgproc_toggles,
                                             imgproc_upscale_toggles, imgproc_realesrgan_model_name, imgproc_sampling,
                                             imgproc_steps, imgproc_height,
                                             imgproc_width, imgproc_cfg, imgproc_denoising, imgproc_seed,
                                             imgproc_gfpgan_strength, imgproc_ldsr_steps, imgproc_ldsr_pre_downSample,
                                             imgproc_ldsr_post_downSample],
                                            [imgproc_output])

                                        imgproc_source.change(
                                            uifn.get_png_nfo,
                                            [imgproc_source],
                                            [imgproc_pngnfo])

                                output_txt2img_to_imglab.click(
                                    fn=uifn.copy_img_params_to_lab,
                                    inputs=[output_txt2img_params],
                                    outputs=[imgproc_prompt, imgproc_seed, imgproc_steps, imgproc_cfg,
                                             imgproc_sampling],
                                )

                                output_txt2img_to_imglab.click(
                                    fn=uifn.copy_img_to_lab,
                                    inputs=[output_txt2img_gallery],
                                    outputs=[imgproc_source, tabs],
                                    _js=call_JS("moveImageFromGallery",
                                                fromId="txt2img_gallery_output",
                                                toId="imglab_input")
                                )
                                if RealESRGAN is None:
                                    with gr.Row():
                                        with gr.Column():
                                            # seperator
                                            gr.HTML("""
        <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
            <p><b> Please download RealESRGAN to activate upscale features</b>, instructions are available at the <a href='https://github.com/hlky/stable-diffusion-webui'>Github</a></p>
        </div>
        """)
            imgproc_toggles.change(fn=uifn.toggle_options_gfpgan, inputs=[imgproc_toggles], outputs=[gfpgan_group])
            imgproc_toggles.change(fn=uifn.toggle_options_upscalers, inputs=[imgproc_toggles], outputs=[upscale_group])
            imgproc_toggles.change(fn=uifn.toggle_options_upscalers, inputs=[imgproc_toggles],
                                   outputs=[upscalerSettings_group])
            imgproc_upscale_toggles.change(fn=uifn.toggle_options_realesrgan, inputs=[imgproc_upscale_toggles],
                                           outputs=[realesrgan_group])
            imgproc_upscale_toggles.change(fn=uifn.toggle_options_ldsr, inputs=[imgproc_upscale_toggles],
                                           outputs=[ldsr_group])
            imgproc_upscale_toggles.change(fn=uifn.toggle_options_gobig, inputs=[imgproc_upscale_toggles],
                                           outputs=[gobig_group])

            """
            if GFPGAN is not None:
                gfpgan_defaults = {
                    'strength': 100,
                }

                if 'gfpgan' in user_defaults:
                    gfpgan_defaults.update(user_defaults['gfpgan'])

                with gr.TabItem("GFPGAN", id='cfpgan_tab'):
                    gr.Markdown("Fix faces on images")
                    with gr.Row():
                        with gr.Column():
                            gfpgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                            gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Effect strength",
                                                        value=gfpgan_defaults['strength'])
                            gfpgan_btn = gr.Button("Generate", variant="primary")
                        with gr.Column():
                            gfpgan_output = gr.Image(label="Output", elem_id='gan_image')
                    gfpgan_btn.click(
                        run_GFPGAN,
                        [gfpgan_source, gfpgan_strength],
                        [gfpgan_output]
                    )
            if RealESRGAN is not None:
                with gr.TabItem("RealESRGAN", id='realesrgan_tab'):
                    gr.Markdown("Upscale images")
                    with gr.Row():
                        with gr.Column():
                            realesrgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                            realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus',
                                                                                                   'RealESRGAN_x4plus_anime_6B'],
                                                                value='RealESRGAN_x4plus')
                            realesrgan_btn = gr.Button("Generate")
                        with gr.Column():
                            realesrgan_output = gr.Image(label="Output", elem_id='gan_image')
                    realesrgan_btn.click(
                        run_RealESRGAN,
                        [realesrgan_source, realesrgan_model_name],
                        [realesrgan_output]
                    )
                output_txt2img_to_upscale_esrgan.click(
                    uifn.copy_img_to_upscale_esrgan,
                    output_txt2img_gallery,
                    [realesrgan_source, tabs],
                    _js=js_move_image('txt2img_gallery_output', 'img2img_editor'))
        """
        gr.HTML("""
    <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
        <p>Stable Diffusion WebUI is an open-source project. You can find the latest builds on the <a href="https://github.com/francislabountyjr/stable-diffusion-webui" target="_blank">main repository</a>.</p>
        <p>This project was forked from <a href="https://github.com/hlky/stable-diffusion-webui" target="_blank">hlky's repository</a>.</p>
    </div>
    """)
        # Hack: Detect the load event on the frontend
        # Won't be needed in the next version of gradio
        # See the relevant PR: https://github.com/gradio-app/gradio/pull/2108
        load_detector = gr.Number(value=0, label="Load Detector", visible=False)
        load_detector.change(None, None, None, _js=js(opt))
        demo.load(lambda x: 42, inputs=load_detector, outputs=load_detector)
    return demo
