// Initial Credits: 
// The Coding Train / Daniel Shiffman
// https://thecodingtrain.com/CodingChallenges/158-shape-classifier.html
// https://youtu.be/3MqJzMvHE3E

let circles = [];
let squares = [];
let triangles = [];
let pentagons = [];
let hexagons = [];
let lenses = [];

// Currently available: 100 | 750 | 500
let samples = "100";
// Currently available: 25 | 50
let percentage = "50";

let numClasses = 6;
let epochs = 35;

function preload() {
  let filename = `data${samples}_warp${percentage}percent`;

  for (let i = 0; i < (parseInt(samples)); i++) {
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
    debug: true,
    layers: [
      { // Add convolutional layers
        type: 'conv2d',
        filters: 16,
        kernelSize: 3,
        activation: 'relu'
      },
      {
        type: 'conv2d',
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
      },
      { // Add pooling layers
        type: 'maxPooling2d',
        poolSize: 2
      },
      { // Flatten the output for fully connected layers
        type: 'flatten'
      },
      { // Add fully connected layers
        type: 'dense',
        units: 64,
        activation: 'relu'
      },
      { // Output layer with the number of classes
        type: 'dense',
        units: numClasses, 
        activation: 'softmax'
      },
    ]
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
  shapeClassifier.train({ epochs: epochs }, finishedTraining);
}

function finishedTraining() {
  console.log('finished training!');
  shapeClassifier.save();
}