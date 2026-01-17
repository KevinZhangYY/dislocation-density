from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for

from ..config import AppConfig, load_config
from .processing import encode_png_base64, run_steps_from_path


def create_app(config_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    cfg = load_config(Path(config_path)) if config_path else AppConfig()
    # ocr_ready check removed

    @app.after_request
    def _no_cache(resp):
        ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct:
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.context_processor
    def _inject_static_version():
        return {"static_v": "dev"}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/run")
    def run():
        f = request.files.get("file")
        if f is None or f.filename is None or f.filename == "":
            return render_template("index.html", error="请拖入或选择一个TIF文件。"), 400

        roi_id = int(request.form.get("roi_id", "0"))
        thickness_nm = request.form.get("thickness_nm", "").strip()
        manual_nm_per_px = request.form.get("manual_nm_per_px", "").strip()
        manual_scale_bar_nm = request.form.get("manual_scale_bar_nm", "").strip()
        max_dim = int(request.form.get("max_dim", "1200"))

        local_cfg = cfg
        if thickness_nm:
            local_cfg = AppConfig(
                m1=local_cfg.m1,
                m2=local_cfg.m2,
                m3=local_cfg.m3,
                m4=local_cfg.m4,
                m5=local_cfg.m5,
                m6=type(local_cfg.m6)(thickness_nm=float(thickness_nm)),
                m7=local_cfg.m7,
                m8=local_cfg.m8,
            )
        
        m1_updates = {}
        if manual_nm_per_px:
            m1_updates["manual_nm_per_px"] = float(manual_nm_per_px)
        if manual_scale_bar_nm:
            m1_updates["manual_scale_bar_nm"] = float(manual_scale_bar_nm)
            
        if m1_updates:
            local_cfg = AppConfig(
                m1=type(local_cfg.m1)(**{**local_cfg.m1.__dict__, **m1_updates}),
                m2=local_cfg.m2,
                m3=local_cfg.m3,
                m4=local_cfg.m4,
                m5=local_cfg.m5,
                m6=local_cfg.m6,
                m7=local_cfg.m7,
                m8=local_cfg.m8,
            )

        suffix = Path(f.filename).suffix or ".tif"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            f.save(tmp_path)

        try:
            res = run_steps_from_path(str(tmp_path), cfg=local_cfg, roi_id=roi_id)
        except RuntimeError as e:
            msg = str(e)
            hint = None
            if "nm/px calibration failed" in msg:
                if not manual_nm_per_px:
                    hint = "标定失败：请在页面上填写“Manual nm/px”（最推荐），或填写“Scale Bar Value (nm)”以便程序计算。"
            return render_template("index.html", error=hint or msg), 400
        except Exception as e:
            return render_template("index.html", error=f"处理失败：{e!s}"), 400
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        steps = res["steps"]
        images = {
            "raw": encode_png_base64(steps.raw, cmap="gray", max_dim=max_dim),
            "roi": encode_png_base64(steps.roi, cmap="gray", max_dim=max_dim),
            "pre": encode_png_base64(steps.pre, cmap="gray", max_dim=max_dim),
            "enhanced": encode_png_base64(steps.enhanced, cmap="magma", max_dim=max_dim),
            "mask": encode_png_base64(steps.mask, cmap="gray", max_dim=max_dim),
            "skeleton": encode_png_base64(steps.skeleton, cmap="gray", max_dim=max_dim),
            "overlay": encode_png_base64(steps.overlay_rgb, cmap=None, max_dim=max_dim),
        }

        return render_template(
            "result.html",
            filename=f.filename,
            nm_per_px=res["nm_per_px"],
            nm_per_px_source=res["nm_per_px_source"],
            roi_box=res["roi_box_xyxy"],
            roi_id=res["roi_id"],
            roi_count=res["roi_count"],
            length_px=res["length_px"],
            metrics=res["metrics"],
            images=images,
        )

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/demo")
    def demo():
        return redirect(url_for("index"))

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="dislo_density_web")
    parser.add_argument("--config", default=None, help="config.yaml path (optional)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
