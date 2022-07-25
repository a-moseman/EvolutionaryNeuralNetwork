package EvolutionaryNeuralNetwork;

import java.util.ArrayList;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        int[] SHAPE = new int[] {1, 16, 16, 1};
        int POOL = 100; // default = 100 (how many ENNs to train with)
        double STDEV = 0.03; // default = 0.03 (how much change per mutation)
        String ACTIVATION_FUNCTION = "tanh";
        double DROPOUT = 0.01;

        int EPOCHS = 1500;
        int BATCHES = 64; // default = 64 (how many sims to do to get loss per ENN per epoch)
        double HOLDOUT = 0.1; // default = 0.1 (% of ENNs that survive and repop per epoch)

        int TRAINING_DATA_SIZE = 100_000;
        int VALIDATION_DATA_SIZE = 3_333;

        Random random = new Random();
        double[][] X = new double[TRAINING_DATA_SIZE][1];
        double[][] Y = new double[TRAINING_DATA_SIZE][1];
        for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
            X[i][0] = random.nextDouble();
            Y[i][0] = Math.sin(X[i][0] * 2 * Math.PI);
        }

        double[][] valX = new double[VALIDATION_DATA_SIZE][1];
        double[][] valY = new double[VALIDATION_DATA_SIZE][1];
        for (int i = 0; i < VALIDATION_DATA_SIZE; i++) {
            valX[i][0] = random.nextDouble();
            valY[i][0] = Math.sin(valX[i][0] * 2 * Math.PI);
        }


        Trainer trainer = new Trainer(POOL)
                .setLayerParams(SHAPE, STDEV, ACTIVATION_FUNCTION, DROPOUT)
                .build();

        ArrayList<NeuralNetwork> agents = trainer.train(EPOCHS, BATCHES, X, Y, valX, valY, HOLDOUT, true);
    }
}
