package EvolutionaryNeuralNetwork;

public class Layer {
    Node[] nodes;

    private Layer previousLayer;
    private Layer nextLayer;

    public Layer(int size, int nextLayerSize, double stdev, String activationFunction, double dropout) {
        this.nodes = new Node[size];
        for (int j = 0; j < nodes.length; j++) {
            nodes[j] = new Node(nextLayerSize, stdev, activationFunction, dropout);
        }
    }

    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public void activate(double[] input) {
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].activate(input[i]);
        }

        if (nextLayer != null) {
            nextLayer.activate();
        }
    }

    private void activate() {
        double[] inputs = new double[nodes.length];
        for (int i = 0; i < previousLayer.nodes.length; i++) {
            for (int j = 0; j < nodes.length; j++) {
                inputs[j] += previousLayer.nodes[i].getTempValue() * previousLayer.nodes[i].connections[j]; // y * w -> f(x)
            }
        }
        activate(inputs);
    }

    public void reset() {
        for (Node node : nodes) {
            node.reset();
        }
    }

    public void mutate() {
        for (Node node : nodes) {
            node.mutate();
        }
    }
}
