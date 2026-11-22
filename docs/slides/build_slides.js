// Builds docs/slides/waste_controller.pptx (6-8 slides, 4-minute talk) from experiments/report_assets/summary.json.
// Run after `python docs/build_report.py`:   node docs/slides/build_slides.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const EXP = path.join(ROOT, "experiments");
const S = JSON.parse(fs.readFileSync(path.join(EXP, "report_assets", "summary.json"), "utf8"));
const m = S.metrics || {};
const f3 = (v) => (v == null ? "—" : v.toFixed(3));
const f1 = (v) => (v == null ? "—" : v.toFixed(1));

const THEME = {
  name: "Waste Controller",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16302B", lt1: "FFFFFF", dk2: "0B7A5E", lt2: "EEF3F1",
    accent1: "009E73", accent2: "56B4E9", accent3: "E69F00", accent4: "D55E00", accent5: "CC79A7", accent6: "0B7A5E",
    hlink: "0B7A5E", folHlink: "4F6762",
  },
};
const MAT = [ // material, label, hex: Okabe-Ito, same as the app overlay
  ["cardboard", "Cardboard", "E69F00"], ["soft_plastic", "Soft plastic", "56B4E9"],
  ["rigid_plastic", "Rigid plastic", "009E73"], ["metal", "Metal", "D55E00"], ["other", "Other", "CC79A7"],
];
const INK = "16302B", MUTED = "4F6762", WASH = "EEF3F1", LINE = "CFDAD7";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Waste Controller";
pres.author = S.author;
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "Title dark",
  background: { color: INK },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.25, w: 8.8, h: 1.4, fontSize: 44, bold: true, color: "FFFFFF", fontFace: "Cambria", valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.8, w: 8.0, h: 1.3, fontSize: 18, color: "CFE3DD", margin: 0, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Content",
  background: { color: "FFFFFF" },
  margin: [0.4, 0.5, 0.4, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 26, bold: true, color: INK, fontFace: "Cambria", valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: MUTED, fontFace: "Calibri" },
});

function chips(slide, x, y, counts, dark) {
  MAT.forEach(([k, label, hex], i) => {
    const cx = x + i * 1.75;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y, w: 0.22, h: 0.32, fill: { color: hex }, rectRadius: 0.03, line: { color: hex }, objectName: `chip-${k}` });
    const t = counts ? `${label}  ${counts[k]}` : label;
    slide.addText(t, { x: cx + 0.3, y: y - 0.02, w: 1.4, h: 0.36, fontSize: 12, color: dark ? "FFFFFF" : INK, margin: 0, valign: "middle", isTextBox: true });
  });
}

function stat(slide, x, y, w, value, label) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 1.25, fill: { color: WASH }, line: { color: WASH }, objectName: `stat-${label}` });
  slide.addText(value, { x: x + 0.2, y: y + 0.1, w: w - 0.4, h: 0.65, fontSize: 32, bold: true, color: INK, fontFace: "Cambria", margin: 0, isTextBox: true });
  slide.addText(label, { x: x + 0.2, y: y + 0.75, w: w - 0.4, h: 0.4, fontSize: 12, color: MUTED, margin: 0, isTextBox: true });
}

function pic(slide, file, x, y, w, h, alt) {
  if (!fs.existsSync(file)) {
    slide.addText(`[${path.basename(file)} not generated yet]`, { x, y, w, h, fontSize: 12, color: MUTED, italic: true, isTextBox: true, fill: { color: WASH } });
    return;
  }
  slide.addImage({ path: file, x, y, w, h, sizing: { type: "contain", w, h }, altText: alt });
}

// 1. Title
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "Title dark", sectionTitle: "Opening" });
s.addText("Waste Controller", { placeholder: "title" });
s.addText("Finding and outlining waste materials in photos, with a model trained on a laptop CPU", { placeholder: "body" });
chips(s, 0.6, 4.25, null, true);
s.addText(`${S.author}  |  CSAI3027P Deep Learning, PBL Project 07`, { x: 0.6, y: 4.85, w: 8.8, h: 0.35, fontSize: 12, color: "9FBDB5", margin: 0, isTextBox: true });
s.addNotes("Hi, I'm presenting Waste Controller: a prototype that looks at a photo of waste, finds every item, outlines it, and names its material. Everything was trained on my laptop, without a GPU.");

// 2. Problem
pres.addSection({ title: "Problem" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Problem" });
s.addText("Wrong items in a recycling stream lower its value", { placeholder: "title" });
s.addText([
  { text: "Sorting lines need to know what each item is made of, and where it is, to keep streams pure.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "Goal: a working prototype that, for every item in a photo, gives", options: { breakLine: true } },
  { text: "a box around it (detection)", options: { bullet: true, breakLine: true } },
  { text: "its outline (segmentation)", options: { bullet: true, breakLine: true } },
  { text: "its material, one of five classes (classification)", options: { bullet: true } },
], { x: 0.5, y: 1.25, w: 4.6, h: 3.2, fontSize: 16, color: INK, paraSpaceAfter: 6, valign: "top", isTextBox: true });
stat(s, 5.5, 1.35, 4.0, "1 model", "does all three jobs: YOLO26n-seg");
stat(s, 5.5, 2.8, 4.0, "0 GPUs", "trained on a Snapdragon X laptop CPU, 8 GB RAM");
s.addNotes("Recycling plants lose money when the wrong material ends up in a stream. A vision system has to know what each item is and exactly where it is. My goal was one model that does detection, segmentation and material classification together.");

// 3. Data
pres.addSection({ title: "Data" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Data" });
s.addText("1,500 litter photos mapped to five materials", { placeholder: "title" });
const tr = MAT.map(([k]) => (S.counts[k] || {}).train || 0);
s.addChart(pres.charts.BAR, [{ name: "Training objects", labels: MAT.map((x) => x[1]), values: tr }], {
  x: 0.4, y: 1.15, w: 5.4, h: 3.9, barDir: "bar", chartColors: MAT.map((x) => x[2]), varyColors: true,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: INK, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: INK, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", valAxisHidden: true,
  valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  showTitle: true, title: "Objects per material (training split)", titleFontSize: 13, titleColor: MUTED, titleFontFace: "+mn-lt",
});
s.addText([
  { text: "TACO dataset: real photos with hand-drawn outlines", options: { bullet: true, breakLine: true } },
  { text: "60 labels mapped to 5 materials; cigarette butts dropped", options: { bullet: true, breakLine: true } },
  { text: `Near-duplicates removed, then split ${S.images.train} / ${S.images.val} / ${S.images.test} (train / val / test)`, options: { bullet: true, breakLine: true } },
  { text: "Cardboard is the scarcest class", options: { bullet: true } },
], { x: 6.0, y: 1.3, w: 3.6, h: 3.6, fontSize: 15, color: INK, paraSpaceAfter: 8, valign: "top", isTextBox: true });
s.addNotes("I used TACO, about 1,500 photos with hand-drawn outlines. I mapped its 60 labels to five recycling materials, removed near-duplicates so test photos can't leak into training, and split 80/10/10. Cardboard is the rarest class, which shows up later in the results.");

// 4. Model and training
pres.addSection({ title: "Model" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Model" });
s.addText("One pass: box, outline and material for every item", { placeholder: "title" });
const steps = [["Photo", "any size, resized to 416 px"], ["YOLO26n-seg", "2.7 M parameters, COCO-pretrained, fine-tuned"], ["Per item", "box + outline mask + material + confidence"]];
steps.forEach(([h, d], i) => {
  const x = 0.5 + i * 3.1;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.35, w: 2.7, h: 1.35, fill: { color: i === 1 ? INK : WASH }, line: { color: i === 1 ? INK : WASH }, rectRadius: 0.08, objectName: `step-${i}` });
  s.addText(h, { x: x + 0.15, y: 1.45, w: 2.4, h: 0.5, fontSize: 18, bold: true, color: i === 1 ? "FFFFFF" : INK, fontFace: "Cambria", margin: 0, isTextBox: true });
  s.addText(d, { x: x + 0.15, y: 1.95, w: 2.4, h: 0.65, fontSize: 12, color: i === 1 ? "CFE3DD" : MUTED, margin: 0, valign: "top", isTextBox: true });
  if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.76, y: 1.88, w: 0.28, h: 0.3, fill: { color: MUTED }, line: { color: MUTED }, objectName: `arrow-${i}` });
});
stat(s, 0.5, 3.25, 2.85, `${S.epochs}`, "epochs on the laptop CPU");
stat(s, 3.575, 3.25, 2.85, S.train_hours || "—", "total training time");
stat(s, 6.65, 3.25, 2.85, "≈4 min", "per epoch (1,197 photos)");
s.addNotes("The model is YOLO26n-seg: a small one-stage network that predicts boxes, materials and outlines in a single pass, so it runs on a CPU. I started from COCO-pretrained weights and fine-tuned on the waste data at 416 pixels with batch size 8, which fits in 8 GB of RAM.");

// 5. Demo
pres.addSection({ title: "Demo" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Demo" });
s.addText("It outlines and names each item", { placeholder: "title" });
const preds = (S.gallery || []).slice(0, 3); // report's picks: one clear example per material first
[0, 1, 2].forEach((i) => pic(s, path.join(EXP, "report_assets", preds[i] || "pred_missing.jpg"), 0.5 + i * 3.05, 1.2, 2.9, 3.3, "Model prediction on a held-out test photo"));
chips(s, 0.5, 4.75, null, false);
s.addNotes("Here are predictions on test photos the model never saw. Each item gets an outline in its material colour, a box, and a confidence score. [Switch to the live Streamlit app: upload a photo or pick a test photo.]");

// 6. Results
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
s.addText("Results on 150 photos it never saw", { placeholder: "title" });
stat(s, 0.5, 1.25, 2.9, f3(m.mask_map50), "mask mAP@0.5 (outlines)");
stat(s, 0.5, 2.65, 2.9, f3(m.box_map50), "box mAP@0.5 (detection)");
stat(s, 0.5, 4.05, 2.9, S.fps ? `${f1(S.fps)} /s` : "—", "photos per second on CPU");
const ap = MAT.map(([k]) => ((m.per_class_box_ap50_95 || {})[k] || 0));
s.addChart(pres.charts.BAR, [{ name: "AP", labels: MAT.map((x) => x[1]), values: ap }], {
  x: 3.7, y: 1.2, w: 5.9, h: 4.0, barDir: "bar", chartColors: MAT.map((x) => x[2]), varyColors: true,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 11, dataLabelColor: INK, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: INK, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", valAxisHidden: true,
  valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  showTitle: true, title: "Detection AP@0.5:0.95 per material (test)", titleFontSize: 13, titleColor: MUTED, titleFontFace: "+mn-lt",
});
s.addNotes("On 150 held-out photos it reaches these scores. Metal is its best material. The biggest weakness is size: it finds about three in four large items but almost none of the tiny ones like bottle caps, because at 416 pixels they are only a few pixels wide. It runs at about 14 photos per second on a laptop CPU.");

// 7. Fundamentals
pres.addSection({ title: "Fundamentals" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "Fundamentals" });
s.addText("The building blocks, written from scratch in NumPy", { placeholder: "title" });
pic(s, path.join(EXP, "fundamentals", "optimizers.png"), 0.5, 1.2, 4.4, 2.9, "Optimizer comparison");
pic(s, path.join(EXP, "fundamentals", "ablations.png"), 5.1, 1.2, 4.4, 2.9, "Regularisation ablations");
s.addText(`Neurons, MLP with backprop (gradient check ${S.gradcheck.toExponential(1)}), GD/SGD/Momentum/Nesterov, SVD, Hebbian/Oja, competitive, perceptron and Hopfield learning, plus ablations, all unit-tested`, {
  x: 0.5, y: 4.3, w: 9.0, h: 0.75, fontSize: 14, color: INK, valign: "top", isTextBox: true, margin: 0 });
s.addNotes("Behind the detector, I implemented the course fundamentals from scratch in NumPy: backpropagation verified by gradient checking, four optimizers, where momentum escapes the plateau plain SGD gets stuck on, classical learning rules, and ablations that show overfitting and how augmentation fixes it.");

// 8. Limits and next steps
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "Title dark", sectionTitle: "Close" });
s.addText("Ready as a demonstrator, not yet for a sorting line", { placeholder: "title", fontSize: 30 });
s.addText([
  { text: "Trained on outdoor litter, not conveyor belts; small or cluttered items get missed", options: { bullet: true, breakLine: true } },
  { text: "Next: retrain on recycling-plant data (WaRP, ZeroWaste) on a GPU at 640 px", options: { bullet: true, breakLine: true } },
  { text: "Add a stream monitor (LSTM) and synthetic data for rare items (GAN)", options: { bullet: true } },
], { placeholder: "body", fontSize: 16, paraSpaceAfter: 6 });
s.addText("Thank you. Questions?", { x: 0.6, y: 4.85, w: 8.8, h: 0.4, fontSize: 14, color: "9FBDB5", margin: 0, isTextBox: true });
s.addNotes("To sum up: it works end to end and runs in real time on a laptop, but it was trained on outdoor litter, so it's a demonstrator, not a production sorter. Next steps are recycling-plant data on a GPU, plus the stream monitor and synthetic data for rare items. Thank you.");

(async () => {
  const out = path.join(__dirname, "waste_controller.pptx");
  await pres.writeFile({ fileName: out });
  console.log("slides ->", out);
})();
