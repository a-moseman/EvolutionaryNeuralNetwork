package EvolutionaryNeuralNetwork;

import java.util.Arrays;
import java.util.Random;

public class Node {
    private static final Random RANDOM = new Random();

    // double weight; // TODO: DEPRECATE?
    double bias;
    double[] connections;
    double stdev;
    String activationFunction;
    double dropout;

    double tempValue;

    public Node(int size, double stdev, String activationFunction, double dropout) {
        double std = Math.sqrt((double) 2 / size);
        // this.weight = RANDOM.nextGaussian() * std; // TODO: deprecate
        //this.bias = RANDOM.nextGaussian() * std; <- this results in NaNs for some reason
        this.bias = 0;
        this.connections = new double[size];
        this.stdev = stdev;
        this.activationFunction = activationFunction;
        this.dropout = dropout;
        Arrays.fill(connections, RANDOM.nextGaussian() * stdev);
    }

    public void mutate() {
        // weight += RANDOM.nextGaussian() * stdev; // TODO: deprecate
        bias += RANDOM.nextGaussian() * stdev;
        for (int i = 0; i < connections.length; i++) {
            connections[i] += RANDOM.nextGaussian() * stdev;
            if (RANDOM.nextDouble() < dropout) {
                connections[i] = 0;
            }
        }
    }

    public void reset() {
        tempValue = 0;
    }

    public void activate(double x) {
        switch (activationFunction) {
            case "relu":
                tempValue = Math.max(0, x + bias);
                break;
            case "sigmoid":
                tempValue = (double) 1 / (1 + Math.exp(-(x + bias)));
                break;
            case "tanh":
                tempValue = Math.tanh(x + bias);
        }
    }

    public double getTempValue() {
        return tempValue;
    }
}
