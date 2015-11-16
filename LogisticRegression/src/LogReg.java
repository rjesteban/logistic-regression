
import Jama.Matrix;
import MultivariateLinearRegression.FeatureNormalizationValues;
import MultivariateLinearRegression.GradientDescentValues;
import MultivariateLinearRegression.MultivariateLR;
import java.io.FileNotFoundException;
import java.io.IOException;
import util.CostPlot;
import util.DataPlot;
import util.Util;

public class LogReg extends MultivariateLR {

    /**
     * %COSTFUNCTIONREG Compute cost and gradient for logistic regression with
     * regularization % J = COSTFUNCTIONREG(theta, X, y, lambda) computes the
     * cost of using % theta as the parameter for regularized logistic
     * regression and the % gradient of the cost w.r.t. to the parameters.
     */
    CostFunctionValues costFunction(Matrix theta, Matrix X, Matrix y, double lambda) {

        int m = y.getRowDimension();
        //CostFunctionValues cfv = costFunction(theta, X, y);
        int row = theta.getRowDimension();
        int col = theta.getColumnDimension();

        double c = 1.d / m;

        Matrix hyp = sigmoid(X.times(theta));

        Matrix one = Util.ones(m, 1);
        Matrix J = new Matrix(1, 1);

        Matrix JJ = y.uminus().arrayTimes(Util.log(hyp)).minus(one.minus(y).
                arrayTimes(Util.log(one.minus(hyp)))).times(c);

        for (int r = 0; r < JJ.getRowDimension(); r++) {
            J.set(0, 0, J.get(0, 0) + JJ.get(r, 0));
        }

        //--------------------partial derivative-----------------------
        Matrix grad = X.transpose().times(hyp.minus(y)).times(c);

        //============= add regularization term to cost ===============
        Matrix thet = theta.getMatrix(1, theta.getRowDimension() - 1, 0, 0);
        thet = Util.pow(thet, 2).times(lambda / (2 * m));
        Matrix k = new Matrix(1, 1);

        for (int r = 0; r < thet.getRowDimension(); r++) {
            k.set(0, 0, k.get(0, 0) + thet.get(r, 0));
        }
        J.plusEquals(k);
        //============= gradient: add regularization term ===============
        thet = theta.getMatrix(1, theta.getRowDimension() - 1, 0, 0);
        thet.timesEquals(lambda / m);

        for (int r = 1; r < grad.getRowDimension(); r++) {
            grad.set(r, 0, (grad.get(r, 0) + thet.get(r - 1, 0)));
        }

        return new CostFunctionValues(J, grad);
    }

    public GradientDescentValues gradientDescent(Matrix X, Matrix y,
            Matrix theta, double alpha, int numIterations, double lambda) {
        int m = y.getRowDimension();
        Matrix J_history = new Matrix(numIterations, 1);

        for (int iter = 0; iter < numIterations; iter++) {
            CostFunctionValues cfv = costFunction(theta, X, y, lambda);
            J_history.set(iter, 0, cfv.getJ().get(0, 0));
            theta = theta.minus(cfv.getGrad().times(alpha));
        }

        return new GradientDescentValues(theta, J_history);
    }

    /*
     function g = sigmoid(z)
     %SIGMOID Compute sigmoid function
     %   J = SIGMOID(z) computes the sigmoid of z.
     */
    Matrix sigmoid(Matrix z) {
        int row = z.getRowDimension();
        int col = z.getColumnDimension();
        Matrix g = new Matrix(row, col);

        for (int r = 0; r < row; r++) {
            g.set(r, 0, (1 / (1 + Math.exp(-z.get(r, 0)))));
        }
        return g;
    }

    /*
     function p = predict(theta, X)
     %PREDICT Predict whether the label is 0 or 1 using learned logistic 
     %regression parameters theta
     %   p = PREDICT(theta, X) computes the predictions for X using a 
     %   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
     */
    Matrix predict(Matrix theta, Matrix X) {
        int m = X.getRowDimension();
        Matrix p = new Matrix(m, 1);

        Matrix h = sigmoid(X.times(theta));

        for (int i = 0; i < m; i++) {
            if (h.get(i, 0) >= 0.5) {
                p.set(i, 0, 1);
            } else {
                p.set(i, 0, 0);
            }
        }
        return p;

    }

    /**
     * fprintf('Train Accuracy: %f\n', );
     */
    double accuracy(Matrix p, Matrix y) {
        double sum = 0.0;

        for (int i = 0; i < p.getRowDimension(); i++) {
            if (p.get(i, 0) == y.get(i, 0)) {
                sum++;
            }
        }
        return (sum / p.getRowDimension()) * 100;

    }

    public static void main(String[] args) throws FileNotFoundException, IOException {

        LogReg lr = new LogReg();

        Matrix X = Util.data("ex2data1.txt", 1, 2);
        Matrix y = Util.data("ex2data1.txt", 3);

        FeatureNormalizationValues fnv = lr.featureNormalize(X);
        /**
         * plotData(X, y);
         *
         * % Put some labels hold on; % Labels and Legend xlabel('Exam 1 score')
         * ylabel('Exam 2 score')
         *
         * % Specified in plot order legend('Admitted', 'Not admitted')
         */
        //===== JAVA CODE HERE ====

        DataPlot dp = new DataPlot(X, y, "Exam 1 score", "Exam 2 score");
        new Thread(dp).start();

        X = fnv.getX();
        X = Util.insertX0(X);
        Matrix initial_theta = new Matrix(X.getColumnDimension(), 1);

        CostFunctionValues cfv = lr.costFunction(initial_theta, X, y, 0);

        System.out.println("Cost at initial theta (zeroes): " + cfv.getJ().get(0, 0));
        System.out.println("Gradient at initial theta (zeroes):");
        cfv.getGrad().print(1, 3);

        double alpha = 1;
        int num_iters = 1000;

        Matrix theta = new Matrix(initial_theta.getRowDimension(), initial_theta.getColumnDimension());
        GradientDescentValues gdv = lr.gradientDescent(X, y, theta, alpha,
                num_iters, 0);

        System.out.println("Theta computed from gradient descent:");
        gdv.getTheta().print(3, 3);

        Matrix _X = new Matrix(3, 1);
        _X.set(0, 0, 1);
        _X.set(1, 0, ((45 - fnv.getMu().get(0, 0)) / fnv.getSigma().get(0, 0)));
        _X.set(2, 0, ((85 - fnv.getMu().get(0, 1)) / fnv.getSigma().get(0, 1)));

        CostPlot cp = new CostPlot(gdv.getCostHistory().getRowPackedCopy());
        new Thread(cp).start();
        
        Matrix prob = lr.sigmoid(gdv.getTheta().transpose().times(_X));

        System.out.print("For a student with scores 45 and 85, we predict an"
                + " admission probability of ");
        prob.print(3, 3);

        Matrix p = lr.predict(gdv.getTheta(), X);

        System.out.println("Train accuracy: " + lr.accuracy(p, y));
    }
}
