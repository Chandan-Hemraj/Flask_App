const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");
const handler = tfn.io.fileSystem("./path/to/your/model.json");
const model = await tf.loadModel(handler)