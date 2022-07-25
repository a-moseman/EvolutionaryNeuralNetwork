package EvolutionaryNeuralNetwork;

import java.util.Random;

public class NeuralNetwork {
    private Random RANDOM = new Random();

    private Layer[] layers;
    private int[] shape;
    private double stdev;
    private String activationFunction;
    private double dropout;

    /**
     * My Evolutionary Neural Network.
     * TODO: implement adding and removing nodes as part of the mutation process
     * @param shape
     * @param stdev
     * @param activationFunction
     */
    public NeuralNetwork(int[] shape, double stdev, String activationFunction, double dropout) {
        this.shape = shape;
        this.stdev = stdev;
        this.activationFunction = activationFunction;
        this.dropout = dropout;
        // init layers
        layers = new Layer[shape.length];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(shape[i], i < layers.length - 1 ? shape[i + 1] : 0, stdev, activationFunction, dropout);
        }
        for (int i = 1; i < layers.length; i++) {
            layers[i].setPreviousLayer(layers[i - 1]);
        }
        for (int i = 0; i < layers.length - 1; i++) {
            layers[i].setNextLayer(layers[i + 1]);
        }
    }

    public NeuralNetwork getOffspring() {
        NeuralNetwork offspring = copy();
        offspring.mutate();
        return offspring;
    }

    public NeuralNetwork getOffspring(NeuralNetwork other) {
        NeuralNetwork offspring = mate(other);
        offspring.mutate();
        return offspring;
    }

    private NeuralNetwork copy() {
        // TODO: fix (biases not dealt with)
        NeuralNetwork copyNeuralNetwork = new NeuralNetwork(shape, stdev, activationFunction, dropout);
        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i].nodes.length; j++) {
                // copyNeuralNetwork.layers[i].nodes[j].weight = layers[i].nodes[j].weight; // TODO: deprecate?

                for (int k = 0; k < layers[i].nodes[j].connections.length; k++) {
                    copyNeuralNetwork.layers[i].nodes[j].connections[k] = layers[i].nodes[j].connections[k];
                }
            }
        }
        return copyNeuralNetwork;
    }

    private NeuralNetwork mean(NeuralNetwork other) {
        // TODO: fix (biases not dealt with)
        NeuralNetwork meanNeuralNetwork = new NeuralNetwork(shape, stdev, activationFunction, dropout);
        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i].nodes.length; j++) {
                // meanNeuralNetwork.layers[i].nodes[j].weight = (layers[i].nodes[j].weight + other.layers[i].nodes[j].weight) / 2; // TODO: deprecate

                for (int k = 0; k < layers[i].nodes[j].connections.length; k++) {
                    meanNeuralNetwork.layers[i].nodes[j].connections[k] = (layers[i].nodes[j].connections[k] + other.layers[i].nodes[j].connections[k]) / 2;
                }
            }
        }
        return meanNeuralNetwork;
    }

    private NeuralNetwork mate(NeuralNetwork other) {
        NeuralNetwork meanNeuralNetwork = new NeuralNetwork(shape, stdev, activationFunction, dropout);
        for (int i = 0; i < layers.length; i++) {
            for (int j = 0; j < layers[i].nodes.length; j++) {

                // double wW = RANDOM.nextInt(2); // TODO: deprecate
                double bW = RANDOM.nextInt(2);

                // meanNeuralNetwork.layers[i].nodes[j].weight = layers[i].nodes[j].weight * wW + other.layers[i].nodes[j].weight * (1 - wW); // TODO: deprecate
                meanNeuralNetwork.layers[i].nodes[j].bias = layers[i].nodes[j].bias * bW + other.layers[i].nodes[j].bias * (1 - bW);


                for (int k = 0; k < layers[i].nodes[j].connections.length; k++) {
                    double cW = RANDOM.nextInt(2);
                    meanNeuralNetwork.layers[i].nodes[j].connections[k] = layers[i].nodes[j].connections[k] * cW + other.layers[i].nodes[j].connections[k] * (1 - cW);
                }
            }
        }
        return meanNeuralNetwork;
    }

    public void mutate() {
        for (Layer layer : layers) {
            layer.mutate();
        }
    }

    public double[] predict(double[] input) {
        assert input.length == layers[0].nodes.length;

        layers[0].activate(input);

        double[] output = new double[layers[layers.length - 1].nodes.length];

        for (int i = 0; i < layers[layers.length - 1].nodes.length; i++) {
            output[i] = layers[layers.length - 1].nodes[i].tempValue;
        }

        reset();

        return output;
    }

    public void reset() {
        for (Layer layer : layers) {
            layer.reset();
        }
    }

    @Override
    public String toString() {
        //return super.toString();

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            for (int j = 0; j < layer.nodes[0].connections.length; j++) {
                sb.append(layer.nodes[0].connections[j]).append(" ").append(layer.nodes[0].bias + ", ");
            }
            sb.append('\n');
        }
        return sb.toString();
    }
}

