// Shape Classifier (Training)
// Coding Challenge
// The Coding Train / Daniel Shiffman
// https://thecodingtrain.com/CodingChallenges/158-shape-classifier.html
// https://youtu.be/3MqJzMvHE3E

// Generate Dataset: https://github.com/CodingTrain/website/tree/gh-pages/CodingChallenges/CC_158_Shape_Classifier/dataset
// Generate Dataset (port): https://editor.p5js.org/codingtrain/sketches/7leVIzy5l
// Training: https://github.com/CodingTrain/website/tree/gh-pages/CodingChallenges/CC_158_Shape_Classifier/training
// Mouse: https://editor.p5js.org/codingtrain/sketches/JgLVfCS4E
// Webcam: https://editor.p5js.org/codingtrain/sketches/2hZGBkqqq

let circles = [];
let squares = [];
let triangles = [];
let pentagons = [];
let hexagons = [];
let lenses = [];

let useMore = false;
let higherPercentage = false;

function preload() {
  let filename = `data${useMore ? "750" : "500"}_warp${higherPercentage ? "50" : "25"}percent`;

  for (let i = 0; i < (useMore ? 750 : 100/*500*/); i++) {
    let index = nf(i + 1, 4, 0);
    circles[i] = loadImage(`data/${filename}/circle${index}.png`);
    squares[i] = loadImage(`data/${filename}/square${index}.png`);
    triangles[i] = loadImage(`data/${filename}/triangle${index}.png`);
    pentagons[i] = loadImage(`data/${filename}/pentagon${index}.png`);
    hexagons[i] = loadImage(`data/${filename}/hexagon${index}.png`);
    lenses[i] = loadImage(`data/${filename}/lens${index}.png`);
  }
}

let shapeClassifier;

function setup() {
  createCanvas(400, 400);
  //background(0);
  //image(squares[0], 0, 0, width, height);

  let options = {
    inputs: [64, 64, 4],
    task: 'imageClassification',
    debug: true
  };
  shapeClassifier = ml5.neuralNetwork(options);

  for (let i = 0; i < circles.length; i++) {
    shapeClassifier.addData({ image: circles[i] }, { label: 'circle' });
    shapeClassifier.addData({ image: squares[i] }, { label: 'square' });
    shapeClassifier.addData({ image: triangles[i] }, { label: 'triangle' });
    shapeClassifier.addData({ image: pentagons[i] }, { label: 'pentagon' });
    shapeClassifier.addData({ image: hexagons[i] }, { label: 'hexagon' });
    shapeClassifier.addData({ image: lenses[i] }, { label: 'lens' });
  }
  shapeClassifier.normalizeData();
  shapeClassifier.train({ epochs: 90 }, finishedTraining);
}

function finishedTraining() {
  console.log('finished training!');
  shapeClassifier.save();
}