
import Jama.Matrix;
import MultivariateLinearRegression.GradientDescentValues;
import java.io.IOException;
import util.CostPlot;
import util.DataPlot;
import util.Util;

public class RegularizedLogReg extends LogReg {

    /**
     * % MAPFEATURE Feature mapping function to polynomial features % %
     * MAPFEATURE(X1, X2) maps the two input features % to quadratic features
     * used in the regularization exercise. % % Returns a new feature array with
     * more features, comprising of % X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2,
     * etc.. % % Inputs X1, X2 must be the same size %
     */
    Matrix mapFeature(Matrix X1, Matrix X2) {
        int degree = 6;
        Matrix out = Util.ones(X1.getRowDimension(), X1.getColumnDimension());

        for (int i = 1; i <= degree; i++) {
            for (int j = 0; j <= i; j++) {
                out = Util.append(out, Util.pow(X1, (i - j)).arrayTimes(Util.pow(X2, j)));
            }
        }
        return out;
    }

    public static void main(String[] args) throws IOException {

        RegularizedLogReg rlr = new RegularizedLogReg();

        /**
         * %% Load Data % The first two columns contains the X values and the
         * third column % contains the label (y).
         *
         * data = load('ex2data2.txt'); X = data(:, [1, 2]); y = data(:, 3);
         *
         * plotData(X, y);
         *
         * % Put some labels hold on;
         *
         * % Labels and Legend xlabel('Microchip Test 1') ylabel('Microchip Test
         * 2')
         *
         * % Specified in plot order legend('y = 1', 'y = 0')
         */
        //========= JAVA CODE =============
        Matrix X = Util.data("ex2data2.txt", 1, 2);
        Matrix y = Util.data("ex2data2.txt", 3);
        DataPlot dp = new DataPlot(X, y, "Microchip Test 1", "Microchip Test 2");
        new Thread(dp).start();

        //put some labels
        /**
         * % Add Polynomial Features
         *
         * % Note that mapFeature also adds a column of ones for us, so the
         * intercept % term is handled X = mapFeature(X(:,1), X(:,2));
         */
        //========= JAVA CODE =============
        X = rlr.mapFeature(X.getMatrix(0, X.getRowDimension() - 1, 0, 0),
                X.getMatrix(0, X.getRowDimension() - 1, 1, 1));
        /**
         *
         * % Initialize fitting parameters initial_theta = zeros(size(X, 2), 1);
         *
         * % Set regularization parameter lambda to 1 lambda = 1;
         *
         */
        Matrix initial_theta = new Matrix(X.getColumnDimension(), 1);
        double lambda = 1;
        /**
         * Optimizing using GD % Choose some alpha value alpha = 0.01; num_iters
         * = 400;
         *
         * % Init Theta and Run Gradient Descent theta = zeros(3, 1); [theta,
         * J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
         *
         * % Plot the convergence graph figure; plot(1:numel(J_history),
         * J_history, '-b', 'LineWidth', 2); xlabel('Number of iterations');
         * ylabel('Cost J');
         *
         * % Display gradient descent's result fprintf('Theta computed from
         * gradient descent: \n'); fprintf(' %f \n', theta);
         */

        //===== JAVA CODE HERE ====
        double alpha = 1;
        int num_iters = 1000;

        GradientDescentValues gdv = rlr.gradientDescent(X, y, initial_theta, alpha, num_iters, lambda);

        System.out.println("Theta computed from gradient descent:");
        gdv.getTheta().print(6, 3);

        Matrix p = rlr.predict(gdv.getTheta(), X);
        System.out.println("Train Accuracy: " + rlr.accuracy(p, y));
        
        CostPlot cp = new CostPlot(gdv.getCostHistory().getRowPackedCopy());
        new Thread(cp).start();

    }

}
