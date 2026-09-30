"""Gradio UI only. Keep your existing prediction function in app.py.

Replace the old Blocks layout and launch call in app.py with:
    from house_ui import build_app, launch_app
    demo = build_app(predict_from_ui)
    if __name__ == '__main__':
        launch_app(demo)

Put this file beside app.py. Supports Gradio 5 and 6. Default categories
are Ames Housing codes; pass choices=... to retain your existing options.
No backend, currency, model, or prediction is invented by this module.
"""

import html
import inspect
import os
from datetime import date

import gradio as gr


CSS = """
.gradio-container {max-width:1240px !important; margin:auto !important;
 background:var(--body-background-fill); font-family:Arial,sans-serif !important;}
#hero {padding:30px 8px 20px;}
.eyebrow {font-size:11px;letter-spacing:.19em;font-weight:800;color:#16816c;}
#hero h1 {font-size:clamp(32px,5vw,54px);line-height:1.06;letter-spacing:-2px;
 margin:12px 0;color:var(--body-text-color);}
#hero p {max-width:580px;font-size:16px;line-height:1.65;color:var(--body-text-color-subdued);}
.hero-line {display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;}
.badge {border:1px solid #bed9ce;border-radius:30px;padding:8px 14px;
 font-size:12px;color:#16816c;}
#workspace {gap:28px;align-items:flex-start;}
#editor {padding:24px;border:1px solid var(--border-color-primary);border-radius:24px;
 background:var(--block-background-fill);}
#editor .tab-nav {gap:8px;padding-bottom:16px;}
#editor .tab-nav button {border-radius:24px;padding:10px 17px;font-weight:700;}
#editor .tab-nav button.selected {background:#123e34;color:#fff;}
.section-note {font-size:14px;line-height:1.6;color:var(--body-text-color-subdued);margin:8px 0 20px;}
#sidebar {gap:16px;}
.scene-card {background:#123e34;border-radius:24px;overflow:hidden;color:#fff;padding:24px;}
.scene-top {display:flex;justify-content:space-between;gap:12px;font-size:11px;
 letter-spacing:.12em;text-transform:uppercase;color:#bcd7cd;}
.scene-card svg {width:100%;height:210px;display:block;margin:12px 0;}
.scene-name {font-size:23px;letter-spacing:-.5px;font-weight:700;margin:6px 0;}
.scene-sub {font-size:12px;color:#bdd2ca;}
.stats {display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:22px;
 border-top:1px solid #ffffff30;padding-top:18px;}
.stats strong {display:block;font-size:20px;color:#fff;}
.stats span {font-size:11px;color:#c1d8ce;}
#estimate-panel {border-radius:24px;padding:22px;border:1px solid var(--border-color-primary);
 background:var(--block-background-fill);}
#predict {border-radius:14px;min-height:52px;background:#d5f28b !important;
 color:#173e33 !important;border:0;font-size:16px;font-weight:800;}
#prediction {padding:18px 0 4px;}
#prediction h1,#prediction h2,#prediction h3 {color:#16816c;}
#footer {text-align:center;font-size:12px;color:var(--body-text-color-subdued);padding:18px;}
@media(min-width:900px){#sidebar{position:sticky;top:20px;}}
@media(max-width:600px){#editor{padding:12px;}#hero{padding-top:14px;}}
"""

DEFAULT_CHOICES = {
    "neighborhood": ["Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr",
        "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV",
        "Mitchel", "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes",
        "OldTown", "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr",
        "Timber", "Veenker"],
    "house_style": ["1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin",
                    "2.5Unf", "SFoyer", "SLvl"],
    "bldg_type": ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"],
    "kitchen_qual": ["Ex", "Gd", "TA", "Fa", "Po"],
    "exter_qual": ["Ex", "Gd", "TA", "Fa", "Po"],
}


def house_preview(area, beds, baths, cars, year, quality, neighborhood, style):
    """Decorative preview only; never estimates a price."""
    esc = lambda value: html.escape(str(value))
    tall = str(style) in {"2Story", "2.5Fin", "2.5Unf"}
    roof_y = 57 if tall else 88
    body_y = 99 if tall else 130
    windows = ''.join(
        f'<rect x="{x}" y="{body_y + 14}" width="23" height="27" rx="3" '
        f'fill="{"#f4d9aa" if quality >= 7 else "#a7cbc1"}"/>'
        for x in (119, 158, 197)
    )
    garage = ('<rect x="236" y="142" width="72" height="54" rx="4" fill="#abc4b5"/>'
              '<path d="M243 155h58m-58 10h58m-58 10h58m-58 10h58" stroke="#648f7d"/>'
              if cars else '')
    return f'''<div class="scene-card">
      <div class="scene-top"><span>Your property snapshot</span><span>◉ Live</span></div>
      <svg viewBox="0 0 360 240" role="img" aria-label="Illustrative house preview">
        <circle cx="285" cy="45" r="24" fill="#f4c9a9"/>
        <ellipse cx="183" cy="202" rx="155" ry="20" fill="#214f40"/>
        <path d="M26 193V130m-15 20 15-24 17 24m-30 17 13-24 16 24"
              stroke="#81ad87" stroke-width="8" fill="none" stroke-linecap="round"/>
        <rect x="106" y="{body_y}" width="128" height="{196-body_y}" rx="3" fill="#f1ecdd"/>
        <path d="M91 {body_y+3}L170 {roof_y}L249 {body_y+3}Z" fill="#e5b28e"/>
        {windows}{garage}
        <rect x="159" y="163" width="26" height="33" rx="3" fill="#376855"/>
        <circle cx="179" cy="181" r="2" fill="#f4d9aa"/>
        <path d="M161 198l-14 24h57l-20-24" fill="#789885"/>
      </svg>
      <div class="scene-name">{esc(neighborhood)} · {esc(style)}</div>
      <div class="scene-sub">Built {int(year)} · Quality {int(quality)}/10 · Illustration only</div>
      <div class="stats"><div><strong>{int(area):,}</strong><span>sq ft living</span></div>
      <div><strong>{int(beds)} / {int(baths)}</strong><span>beds / full baths</span></div>
      <div><strong>{int(cars)}</strong><span>garage spaces</span></div></div>
    </div>'''


def build_app(predict_from_ui, choices=None):
    """Accept the existing callable unchanged; customize original category values.

    choices example: {'kitchen_qual': ['Excellent', 'Good', 'Typical', 'Fair']}
    All values are passed to your function exactly as selected.
    """
    options = {**DEFAULT_CHOICES, **(choices or {})}
    def dropdown(key, label, preferred):
        values = options[key]
        # Supports Gradio (display label, backend value) tuples too.
        raw = [v[1] if isinstance(v, (tuple, list)) else v for v in values]
        if not raw:
            raise ValueError(f"Provide at least one choice for {key}")
        return gr.Dropdown(values, value=preferred if preferred in raw else raw[0],
                           label=label, allow_custom_value=False)

    theme = gr.themes.Soft(primary_hue="emerald", secondary_hue="orange",
                           neutral_hue="stone", font=["Arial", "sans-serif"]).set(
        body_background_fill="#f7f6f0", block_background_fill="#ffffff",
        button_primary_background_fill="#123e34",
        button_primary_text_color="#ffffff",
    )
    # Gradio 6 moved theme/CSS from Blocks to launch; support either version.
    styling = {"theme": theme, "css": CSS}
    old_api = "theme" in inspect.signature(gr.Blocks.__init__).parameters
    with gr.Blocks(title="House Price Predictor", **(styling if old_api else {})) as demo:
        gr.HTML('''<header id="hero"><div class="hero-line">
          <span class="eyebrow">HOUSE PRICE PREDICTOR / PROPERTY STUDIO</span>
          <span class="badge">Your next chapter starts here</span></div>
          <h1>A little detail.<br>A clearer picture.</h1>
          <p>Explore the home you have in mind. Add its spaces, finishes, and location,
          then let your prediction model do the numbers.</p></header>''')
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=7, min_width=320, elem_id="editor"):
                with gr.Tabs():
                    with gr.Tab("01 · Spaces"):
                        gr.HTML('<p class="section-note">Start with the footprint. Picture the spaces that make this house a home.</p>')
                        living_area = gr.Slider(100, 10000, value=1500, step=10, label="Living area · sq ft")
                        with gr.Row():
                            lot_area = gr.Number(value=8000, minimum=0, precision=0, label="Lot area · sq ft")
                            total_bsmt_sf = gr.Number(value=800, minimum=0, precision=0, label="Basement area · sq ft")
                        year_built = gr.Slider(1800, date.today().year, value=2000, step=1, label="Year built")
                        gr.Markdown("### Room to live")
                        with gr.Row():
                            bedrooms = gr.Slider(0, 10, value=3, step=1, label="Bedrooms")
                            full_bath = gr.Slider(0, 8, value=2, step=1, label="Full bathrooms")
                        fireplaces = gr.Slider(0, 5, value=1, step=1, label="Fireplaces")
                    with gr.Tab("02 · Finishes"):
                        gr.HTML('<p class="section-note">The details make the difference. Add finish quality and garage space.</p>')
                        overall_qual = gr.Slider(1, 10, value=7, step=1, label="Overall quality", info="1 = very poor · 10 = excellent")
                        with gr.Row():
                            kitchen_qual = dropdown("kitchen_qual", "Kitchen quality", "Gd")
                            exter_qual = dropdown("exter_qual", "Exterior quality", "Gd")
                        gr.Markdown("### A place to park")
                        garage_cars = gr.Slider(0, 5, value=2, step=1, label="Garage capacity · cars")
                        garage_area = gr.Number(value=450, minimum=0, precision=0, label="Garage area · sq ft")
                    with gr.Tab("03 · Setting"):
                        gr.HTML('<p class="section-note">Find its place. Select the neighborhood and architectural style.</p>')
                        neighborhood = dropdown("neighborhood", "Neighborhood", "NAmes")
                        house_style = dropdown("house_style", "House style", "1Story")
                        bldg_type = dropdown("bldg_type", "Building type", "1Fam")
                        gr.Markdown("Ready? Review your property snapshot, then select **Predict price**.")
            with gr.Column(scale=4, min_width=300, elem_id="sidebar"):
                preview = gr.HTML(house_preview(1500, 3, 2, 2, 2000, 7,
                                                neighborhood.value, house_style.value))
                with gr.Column(elem_id="estimate-panel"):
                    gr.Markdown("### Your price outlook\nAn estimate based on your property's details.")
                    predict_button = gr.Button("Predict price ↗", variant="primary", elem_id="predict")
                    status = gr.Markdown("Adjust the details, then generate your estimate.")
                    with gr.Column(visible=False, elem_id="prediction") as result_card:
                        result = gr.Markdown()
                        gr.Markdown("*Model estimate, not a professional appraisal.*")
        gr.HTML('<div id="footer">Make room for possibilities. · House Price Predictor</div>')

        # Exact positional order required by your existing function.
        inputs = [living_area, overall_qual, year_built, garage_cars, garage_area,
                  total_bsmt_sf, bedrooms, full_bath, neighborhood, house_style,
                  bldg_type, kitchen_qual, exter_qual, lot_area, fireplaces]
        snapshot_inputs = [living_area, bedrooms, full_bath, garage_cars,
                           year_built, overall_qual, neighborhood, house_style]
        def refresh(*values):
            return (house_preview(*values), "Details updated. Select **Predict price** for a new estimate.",
                    gr.update(visible=False))
        # Shared queue keeps edits and predictions ordered, avoiding stale cards.
        gr.on(triggers=[component.change for component in inputs], fn=refresh,
              inputs=snapshot_inputs, outputs=[preview, status, result_card],
              concurrency_id="property-ui", concurrency_limit=1,
              trigger_mode="always_last", show_progress="hidden")
        event = predict_button.click(fn=predict_from_ui, inputs=inputs,
              outputs=[result, result_card], concurrency_id="property-ui",
              concurrency_limit=1, trigger_mode="once")
        event.success(lambda: "Estimate ready for the submitted details.", outputs=status)
    demo._house_launch_options = {} if old_api else styling
    return demo


def launch_app(demo):
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
        share=False,
        **demo._house_launch_options,
    )
