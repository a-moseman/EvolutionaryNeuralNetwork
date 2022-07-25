package EvolutionaryNeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.Random;

public class Trainer {
    private static Random RANDOM = new Random();
    public static double avgLoss;
    public static double avgValLoss;

    private int pool;
    private int[] shape;
    private double stdev;
    private String activationFunction;
    private double dropout;

    private ArrayList<NeuralNetwork> agents;
    private ArrayList<Double> losses;
    private ArrayList<Double> validationLosses;


    public Trainer(int pool) {
        this.pool = pool;
    }

    public Trainer setLayerParams(int[] shape, double stdev, String activationFunction, double dropout) {
        this.shape = shape;
        this.stdev = stdev;
        this.activationFunction = activationFunction;
        this.dropout = dropout;
        return this;
    }

    public Trainer build() {
        agents = new ArrayList<>();
        for (int i = 0; i < pool; i++) {
            agents.add(new NeuralNetwork(shape, stdev, activationFunction, dropout));
        }
        return this;
    }

    public ArrayList<NeuralNetwork> train(int epochs, int simulationsPerEpoch, double[][] x, double[][] y, double[][] valX, double[][] valY, double holdout, boolean visualize) {
        losses = new ArrayList<>();
        validationLosses = new ArrayList<>();


        JFrame jFrame = new JFrame();
        View view = new View(epochs);
        if (visualize) {
            jFrame.add(view);
            view.setPreferredSize(new Dimension(epochs, 640));
            jFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            jFrame.pack();
            jFrame.setVisible(true);
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            if (epoch > 0) {
                purge(holdout);
                repopulate();
            }

            // test loss
            for (int i = 0; i < pool; i++) {
                double loss = simulate(agents.get(i), simulationsPerEpoch, x, y);
                losses.add(loss);
                double valLoss = simulate(agents.get(i), simulationsPerEpoch, valX, valY);
                validationLosses.add(valLoss);
            }
            avgLoss = 0;
            for (double loss : losses) {
                avgLoss += loss;
            }
            avgLoss = avgLoss / losses.size();

            avgValLoss = 0;
            for (double loss : validationLosses) {
                avgValLoss += loss;
            }
            avgValLoss = avgValLoss / validationLosses.size();

            System.out.println("Epoch " + epoch + ": avg loss = " + avgLoss + " | avg val loss: " + avgValLoss);

            if (visualize) {
                view.repaint();
            }
        }
        return agents;
    }

    private void repopulate() {
        ArrayList<NeuralNetwork> newAgents = new ArrayList<>();

        while (newAgents.size() < pool) {
            NeuralNetwork offspring = agents.get(RANDOM.nextInt(agents.size())).getOffspring(agents.get(RANDOM.nextInt(agents.size())));
            newAgents.add(offspring);
        }

        agents = newAgents;
    }

    private void purge(double holdout) {
        while (agents.size() > pool * holdout) {
            double worstLoss = losses.get(0);
            int worstIndex = 0;

            for (int j = 0; j < agents.size(); j++) {
                if (losses.get(j) < worstLoss) {
                    worstLoss = losses.get(j);
                    worstIndex = j;
                }
            }
            agents.remove(worstIndex);
            losses.remove(worstIndex);
        }
        losses.clear();
    }

    private double simulate(NeuralNetwork agent, int simulationsPerEpoch, double[][] x, double[][] y) {
        double loss = 0;
        for (int i = 0; i < simulationsPerEpoch; i++) {
            int choice = RANDOM.nextInt(x.length);
            double[] estimate = agent.predict(x[choice]);
            double[] trueY = y[choice];
            loss += lossFunction(estimate, trueY);
        }
        loss = loss / simulationsPerEpoch;
        return -loss;
    }

    private double lossFunction(double[] est, double[] tru) {
        double loss = 0;
        for (int i = 0; i < est.length; i++) {
            loss += Math.pow(tru[i] - est[i], 2);
        }
        return loss / est.length;
    }
}
