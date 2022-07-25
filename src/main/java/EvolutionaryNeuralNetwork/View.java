package EvolutionaryNeuralNetwork;

import javax.swing.*;
import java.awt.*;

public class View extends JPanel {

    private double[] avgLosses;
    private double[] avgValLosses;
    private int size;

    public View(int epochs) {
        this.avgLosses = new double[epochs];
        this.avgValLosses = new double[epochs];
    }

    @Override
    public void paint(Graphics g) {
        if (avgLosses.length > size) {
            avgLosses[size] = Trainer.avgLoss;
            avgValLosses[size] = Trainer.avgValLoss;
            size++;
        }
        super.paint(g);
        for (int i = 0; i < size; i++) {
            if (i > 0) {
                // draw loss
                g.setColor(Color.BLUE);
                g.drawLine(i - 1, (int)((avgLosses[i - 1] + 1) * 640), i, (int)((avgLosses[i] + 1) * 640));
                // draw validation loss
                g.setColor(Color.RED);
                g.drawLine(i - 1, (int)((avgValLosses[i - 1] + 1) * 640), i, (int)((avgValLosses[i] + 1) * 640));
            }
        }
    }
}