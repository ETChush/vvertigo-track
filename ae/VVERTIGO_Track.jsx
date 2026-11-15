/*
 * VVERTIGO Track Importer  v1.0
 * 把 run.py 导出的 A-ID*.csv 写成 After Effects 的 Position / Scale 关键帧。
 *
 * 安装：放到任意位置，AE 菜单 文件 > 脚本 > 运行脚本文件… 选中本文件即可。
 *       常用的话，把本文件复制到
 *         Windows: C:\Program Files\Adobe\Adobe After Effects <版本>\Support Files\Scripts\ScriptUI Panels\
 *       然后在 AE 的「窗口」菜单里就能直接打开成常驻面板。
 *
 * 用法：
 *   1. 打开（或新建）一个 1080x1920 的合成，把原片拖进去，选中该图层
 *   2. 运行本脚本，选择 A-ID1.csv，点「导入」
 *   3. 图层的 Position（以及 Scale）就带上了取景动画
 *
 * 两种模式：
 *   稀疏关键帧（默认）：只打 CSV 里 kf=1 的那些点，带贝塞尔手柄。可以手动微调，
 *                       曲线和成片一致，误差写在 CSV 头的 max_error_px 里。
 *   逐帧烘焙：每帧一个关键帧，像素级精确，但关键帧多、不方便手改。
 *
 * 坐标系换算（脚本内部做的事）：
 *   CSV 给的是「裁切框中心」的源视频像素坐标。要把源片放进 comp，需要
 *     scale  = comp.width / crop_w
 *     pos.x  = comp.width/2  - (x - src_w/2) * scale
 *     pos.y  = comp.height/2 - (y - src_h/2) * scale
 *   前提：图层锚点在素材中心（拖进来的素材默认就是），且没有父级缩放。
 */

(function (thisObj) {

    var SCRIPT_NAME = "VVERTIGO 点位导入";
    var VERSION = "1.0";

    // ------------------------------------------------------------ 读取与解析
    function readText(path) {
        var f = new File(path);
        if (!f.exists) {
            return null;
        }
        f.encoding = "UTF-8";
        if (!f.open("r")) {
            return null;
        }
        var s = f.read();
        f.close();
        return s;
    }

    function parsePoints(path) {
        var text = readText(path);
        if (text === null) {
            return { error: "读不到文件：" + path };
        }
        var lines = text.split(/\r\n|\r|\n/);
        var meta = {};
        var idx = {};
        var rows = [];
        var haveHeader = false;
        var i, t, eq, toks, parts;
        for (i = 0; i < lines.length; i++) {
            var line = lines[i];
            if (!line) {
                continue;
            }
            if (line.charAt(0) === "#") {
                toks = line.substring(1).replace(/^\s+/, "").split(/\s+/);
                for (t = 0; t < toks.length; t++) {
                    eq = toks[t].indexOf("=");
                    if (eq > 0) {
                        meta[toks[t].substring(0, eq)] = toks[t].substring(eq + 1);
                    }
                }
                continue;
            }
            if (!haveHeader) {
                parts = line.split(",");
                for (t = 0; t < parts.length; t++) {
                    idx[parts[t]] = t;
                }
                haveHeader = true;
                continue;
            }
            parts = line.split(",");
            if (parts.length >= 6) {
                rows.push(parts);
            }
        }
        if (!rows.length) {
            return { error: "文件里没有数据行" };
        }
        if (idx.frame === undefined || idx.x === undefined) {
            return { error: "缺少 frame / x 列，可能不是 run.py 导出的文件" };
        }
        return { meta: meta, idx: idx, rows: rows };
    }

    function num(row, idx, name, def) {
        var i = idx[name];
        if (i === undefined) {
            return def;
        }
        var v = parseFloat(row[i]);
        return isNaN(v) ? def : v;
    }

    function fmt(v, d) {
        var p = Math.pow(10, d);
        return Math.round(v * p) / p;
    }

    // ------------------------------------------------------------ 打关键帧
    function setScaleValue(prop, s) {
        var v = prop.value;
        var arr = [];
        for (var i = 0; i < v.length; i++) {
            arr.push(s);
        }
        if (v.length === 3) {
            arr[2] = v[2];
        }
        prop.setValue(arr);
    }

    function isSeparated(prop) {
        return prop.propertyType === PropertyType.INDEXED_GROUP ||
               prop.propertyType === PropertyType.NAMED_GROUP;
    }

    function applyTrack(comp, layer, data, mode) {
        var meta = data.meta, idx = data.idx, rows = data.rows;
        var srcW = parseFloat(meta.width);
        var srcH = parseFloat(meta.height);
        var fps = parseFloat(meta.fps) || comp.frameRate;
        var cw = num(rows[0], idx, "crop_w", 0);
        if (!cw) {
            cw = comp.width;
        }
        var scale = comp.width / cw;

        var posProp = layer.property("ADBE Transform Group").property("ADBE Position");
        var scaleProp = layer.property("ADBE Transform Group").property("ADBE Scale");
        setScaleValue(scaleProp, scale * 100);

        // 选哪些行
        var use = [];
        for (var r = 0; r < rows.length; r++) {
            if (mode === "baked" || num(rows[r], idx, "kf", 0) === 1) {
                use.push(r);
            }
        }
        if (use.length < 2) {
            return { error: "关键帧不足 2 个（CSV 可能被截断）" };
        }

        // 预算每帧的 comp 坐标
        var times = [], px = [], py = [];
        for (var k = 0; k < use.length; k++) {
            var row = rows[use[k]];
            var t = num(row, idx, "frame", 0) / fps;
            var x = num(row, idx, "x", srcW / 2);
            var y = num(row, idx, "y", srcH / 2);
            times.push(t);
            px.push(comp.width / 2 - (x - srcW / 2) * scale);
            py.push(comp.height / 2 - (y - srcH / 2) * scale);
        }

        var separated = isSeparated(posProp);
        var dim = 2;
        var i, d;

        if (separated) {
            // Position 被拆成了 X / Y 两个独立属性，只能分别打点
            var xProp = posProp.property(1);
            var yProp = posProp.property(2);
            for (k = 0; k < times.length; k++) {
                xProp.setValueAtTime(times[k], px[k]);
                yProp.setValueAtTime(times[k], py[k]);
            }
            for (k = 0; k < times.length; k++) {
                xProp.setInterpolationTypeAtKey(k + 1, KeyframeInterpolationType.LINEAR, KeyframeInterpolationType.LINEAR);
                yProp.setInterpolationTypeAtKey(k + 1, KeyframeInterpolationType.LINEAR, KeyframeInterpolationType.LINEAR);
            }
        } else {
            var cur = posProp.value;
            dim = cur.length;
            var vals = [];
            for (k = 0; k < times.length; k++) {
                var one = [];
                for (d = 0; d < dim; d++) {
                    one.push(d === 0 ? px[k] : (d === 1 ? py[k] : cur[d]));
                }
                vals.push(one);
            }
            posProp.setValuesAtTimes(times, vals);
            for (k = 0; k < times.length; k++) {
                posProp.setInterpolationTypeAtKey(k + 1, KeyframeInterpolationType.LINEAR, KeyframeInterpolationType.LINEAR);
            }

            // 空间贝塞尔手柄：入切线朝前一段（负），出切线朝后一段（正）
            if (mode !== "baked") {
                var nOT = 0;
                for (k = 0; k < use.length; k++) {
                    var row2 = rows[use[k]];
                    if (num(row2, idx, "kf", 0) !== 1) {
                        continue;
                    }
                    var inT = [num(row2, idx, "tix", 0) * scale,
                               num(row2, idx, "tiy", 0) * scale];
                    var outT = [num(row2, idx, "tox", 0) * scale,
                                num(row2, idx, "toy", 0) * scale];
                    if (dim === 3) {
                        inT.push(0);
                        outT.push(0);
                    }
                    try {
                        posProp.setSpatialTangentsAtKey(nOT + 1, inT, outT);
                    } catch (e) {
                        // 老版本 AE 没有这个方法，退化成直线插值即可
                    }
                    nOT++;
                }
            }
        }

        return {
            keys: use.length,
            totalFrames: rows.length,
            scale: scale,
            cropW: cw,
            separated: separated,
            dim: dim,
            duration: times[times.length - 1]
        };
    }

    // ------------------------------------------------------------ UI
    function buildUI(thisObj) {
        var win = (thisObj instanceof Panel)
            ? thisObj
            : new Window("dialog", SCRIPT_NAME + "  " + VERSION, undefined, { resizeable: true });

        win.orientation = "column";
        win.alignChildren = ["fill", "top"];
        win.spacing = 10;
        win.margins = 16;

        var rowFile = win.add("group");
        rowFile.orientation = "row";
        rowFile.alignChildren = ["left", "center"];
        rowFile.add("statictext", undefined, "点位 CSV");
        var editPath = rowFile.add("edittext", undefined, "");
        editPath.characters = 34;
        var btnBrowse = rowFile.add("button", undefined, "浏览…");

        var infoBox = win.add("panel", undefined, "文件信息");
        infoBox.orientation = "column";
        infoBox.alignChildren = ["fill", "top"];
        infoBox.margins = 12;
        var infoText = infoBox.add("statictext", undefined, "（还没选文件）");
        infoText.preferredSize = [430, 92];

        var rowMode = win.add("group");
        rowMode.orientation = "row";
        rowMode.add("statictext", undefined, "导入方式");
        var ddMode = rowMode.add("dropdownlist", undefined,
            ["稀疏关键帧（可编辑，推荐）", "逐帧烘焙（像素级精确）"]);
        ddMode.selection = 0;

        var rowTarget = win.add("group");
        rowTarget.orientation = "row";
        rowTarget.add("statictext", undefined, "打在哪");
        var ddTarget = rowTarget.add("dropdownlist", undefined,
            ["当前选中图层", "新建空对象（合成中心）"]);
        ddTarget.selection = 0;

        var hint = win.add("statictext", undefined,
            "提示：合成的尺寸应为 1080×1920，图层锚点在素材中心；" +
            "若合成帧率与素材不同，关键帧仍按真实时间落点。");
        hint.preferredSize = [430, 46];

        var rowBtn = win.add("group");
        rowBtn.alignment = ["right", "top"];
        var btnRun = rowBtn.add("button", undefined, "导入");
        var btnClose = rowBtn.add("button", undefined, "关闭");

        var state = { data: null, path: "" };

        function refreshInfo() {
            if (!state.data || state.data.error) {
                infoText.text = state.data ? state.data.error : "（还没选文件）";
                return;
            }
            var m = state.data.meta, r = state.data.rows;
            var kf = 0, i;
            for (i = 0; i < r.length; i++) {
                if (num(r[i], state.data.idx, "kf", 0) === 1) {
                    kf++;
                }
            }
            var cw = num(r[0], state.data.idx, "crop_w", 0);
            var comp = app.project.activeItem;
            var compW = (comp && comp instanceof CompItem) ? comp.width : 1080;
            var sc = compW / (cw || compW);
            var lines = [];
            lines.push("源 " + m.width + "×" + m.height + "  @" + m.fps + "fps  "
                + r.length + " 帧");
            lines.push("裁切 " + cw + "×" + num(r[0], state.data.idx, "crop_h", 0)
                + "  →  缩放 " + fmt(sc * 100, 1) + "%");
            lines.push("目标 " + (m.target || "-") + (m.gid ? "（gid " + m.gid + "）" : "")
                + "  覆盖 " + (m.coverage ? fmt(parseFloat(m.coverage) * 100, 1) + "%" : "-"));
            lines.push("关键帧 " + kf + "（每 " + m.frames_per_keyframe + " 帧）"
                + "  曲线误差 " + m.max_error_px + "px");
            infoText.text = lines.join("\n");
        }

        btnBrowse.onClick = function () {
            var f = File.openDialog("选择点位 CSV", "CSV:*.csv,所有文件:*.*");
            if (!f) {
                return;
            }
            state.path = f.fsName;
            editPath.text = decodeURI(f.name);
            state.data = parsePoints(state.path);
            refreshInfo();
        };

        btnRun.onClick = function () {
            if (!state.data || state.data.error) {
                alert("先选一个有效的 CSV 文件。");
                return;
            }
            var comp = app.project.activeItem;
            if (!comp || !(comp instanceof CompItem)) {
                alert("请先激活一个合成（时间线面板）。");
                return;
            }

            var srcFps = parseFloat(state.data.meta.fps) || comp.frameRate;
            var warn = [];
            if (Math.abs(srcFps - comp.frameRate) > 0.01) {
                warn.push("合成帧率 " + fmt(comp.frameRate, 2) + "fps 与素材 "
                    + fmt(srcFps, 2) + "fps 不同，关键帧按真实时间落点，可能不在整帧上。");
            }
            if (Math.abs(comp.width / comp.height - 9 / 16) > 0.01) {
                warn.push("合成不是 9:16（当前 " + comp.width + "×" + comp.height + "），取景比例会不同。");
            }

            var mode = (ddMode.selection.index === 1) ? "baked" : "bezier";

            app.beginUndoGroup(SCRIPT_NAME);
            var layer;
            if (ddTarget.selection.index === 1) {
                layer = comp.layers.addNull();
                layer.name = "VVERTIGO_CAM";
                layer.property("ADBE Transform Group").property("ADBE Position")
                    .setValue([comp.width / 2, comp.height / 2]);
            } else {
                layer = comp.selectedLayers[0];
                if (!layer) {
                    app.endUndoGroup();
                    alert("先选中一个图层（通常是原片），或者把「打在哪」改成新建空对象。");
                    return;
                }
            }

            var res;
            try {
                res = applyTrack(comp, layer, state.data, mode);
            } catch (err) {
                app.endUndoGroup();
                alert("导入失败：" + err.toString());
                return;
            }
            app.endUndoGroup();

            if (res.error) {
                alert(res.error);
                return;
            }
            if (res.keys > 4000 && mode === "baked") {
                warn.push("烘焙了 " + res.keys + " 个关键帧，图层会比较重。可以改用稀疏模式，或全选关键帧后右键「关键帧辅助 > 简化」。");
            }
            if (res.separated) {
                warn.push("该图层的 Position 是分离维度（X/Y 分开），已分别打点，无法写入空间贝塞尔手柄。");
            }
            warn.push("完成：" + res.keys + " / " + res.totalFrames + " 帧打点，缩放 "
                + fmt(res.scale * 100, 1) + "%，时长 " + fmt(res.duration, 2) + "s。");
            alert(warn.join("\n\n"));
            if (thisObj instanceof Panel) {
                win.layout.layout(true);
            }
        };

        btnClose.onClick = function () {
            if (win instanceof Window) {
                win.close();
            }
        };

        if (win instanceof Window) {
            win.center();
            win.show();
        } else {
            win.layout.layout(true);
            win.onResizing = win.onResize = function () {
                this.layout.resize();
            };
        }
        return win;
    }

    buildUI(thisObj);

})(this);
