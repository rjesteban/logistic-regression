
import Jama.Matrix;
import MultivariateLinearRegression.FeatureNormalizationValues;
import MultivariateLinearRegression.GradientDescentValues;
import MultivariateLinearRegression.MultivariateLR;
import java.io.FileNotFoundException;
import java.io.IOException;
import util.DataPlot;
import util.Util;

public class LogReg extends MultivariateLR{
	


	/**
     * GRADIENTDESCENTMULTI Performs gradient descent to learn theta
        theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
        taking num_iters gradient steps with learning rate alpha
     * @param X
     * @param y
     * @param theta
     * @param alpha
     * @param numIterations
     * @return 
     */
    @Override
    public GradientDescentValues gradientDescent(Matrix X, Matrix y, Matrix theta, double alpha, int numIterations){
        int m = y.getRowDimension();
        Matrix J_history = new Matrix(numIterations,1);
        
        
        
        for(int iter=0;iter<numIterations;iter++){
            CostFunctionValues cfv = costFunction(theta, X, y);
            J_history.set(iter, 0, cfv.getJ().get(0, 0));
            theta = theta.minus(cfv.getGrad().times(alpha));
        }
        return new GradientDescentValues(theta, J_history);
    }

    /**
		%COSTFUNCTION Compute cost and gradient for logistic regression
		%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
		%   parameter for logistic regression and the gradient of the cost
		%   w.r.t. to the parameters.
	*/
    double costFunctionGD(Matrix theta, Matrix X, Matrix y){
        CostFunctionValues cfv = costFunction(theta, X, y);
        return cfv.getJ().get(0, 0);

    }


	/**
		%COSTFUNCTION Compute cost and gradient for logistic regression
		%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
		%   parameter for logistic regression and the gradient of the cost
		%   w.r.t. to the parameters.
	*/
    CostFunctionValues costFunction(Matrix theta, Matrix X, Matrix y){
        int m = y.getRowDimension();
        int row = theta.getRowDimension();
        int col = theta.getColumnDimension();


        double c = 1.d/m;

        Matrix hyp = sigmoid(X.times(theta));

        Matrix one = Util.ones(m, 1);
        Matrix J = new Matrix(1,1);

        Matrix JJ = y.uminus().arrayTimes(log(hyp)).minus(one.minus(y).
                arrayTimes(log(one.minus(hyp)))).times(c);

        for(int r=0; r<JJ.getRowDimension();r++)
            J.set(0, 0, J.get(0, 0) + JJ.get(r, 0));

        //--------------------partial derivative-----------------------
        Matrix grad = X.transpose().times(hyp.minus(y)).times(c);

        return new CostFunctionValues(J, grad);

    }
        
    Matrix log(Matrix hyp){
        int row = hyp.getRowDimension();
        int col = hyp.getColumnDimension();
        Matrix j = new Matrix(row, col);
        for(int r=0; r<row; r++){
            j.set(r, 0, Math.log(hyp.get(r, 0)));
        }

        return j;
    }

    /*
    function g = sigmoid(z)
    %SIGMOID Compute sigmoid function
    %   J = SIGMOID(z) computes the sigmoid of z.
    */
    Matrix sigmoid(Matrix z){
        int row = z.getRowDimension();
        int col = z.getColumnDimension();
        Matrix g = new Matrix(row, col);

        for(int r=0; r<row; r++){
            g.set(r, 0, (1/(1+Math.exp(-z.get(r, 0)))));
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
    Matrix predict(Matrix theta, Matrix X){
        int m = X.getRowDimension();
        Matrix p = new Matrix(m,1);

        Matrix h = sigmoid(X.times(theta));

        for(int i=0 ; i<m; i++){
            if(h.get(i,0)>=0.5)
                p.set(i, 0, 1);
            else
                p.set(i, 0, 0);
        }
        return p;

    }


    /**
            fprintf('Train Accuracy: %f\n', );		
    */
    double accuracy(Matrix p, Matrix y){
        double sum = 0.0;

        for(int i=0; i<p.getRowDimension(); i++){
            if(p.get(i,0)==y.get(i,0))
                sum++;
        }
        return (sum/p.getRowDimension())*100;

    }

    

    public static void main(String[] args) throws FileNotFoundException, IOException{

        LogReg lr = new LogReg();

        Matrix X = Util.data("ex2data1.txt", 1, 2);
        Matrix y = Util.data("ex2data1.txt", 3);		

        FeatureNormalizationValues fnv = lr.featureNormalize(X);
            /**
                    plotData(X, y);

                    % Put some labels 
                    hold on;
                    % Labels and Legend
                    xlabel('Exam 1 score')
                    ylabel('Exam 2 score')

                    % Specified in plot order
                    legend('Admitted', 'Not admitted')
            */
            //===== JAVA CODE HERE ====
        
            DataPlot dp = new DataPlot(X, y,"Exam 1 score", "Exam 2 score");  
            new Thread(dp).start();

        X = fnv.getX();
        X = Util.insertX0(X);
        Matrix initial_theta = new Matrix(X.getColumnDimension(), 1);

        CostFunctionValues cfv = lr.costFunction(initial_theta, X, y);

        System.out.println("Cost at initial theta (zeroes): " + cfv.getJ().get(0,0));
        System.out.println("Gradient at initial theta (zeroes):");
        cfv.getGrad().print(1, 3);

        double alpha = 1;
        int num_iters = 1000;

        Matrix theta = new Matrix(initial_theta.getRowDimension(),initial_theta.getColumnDimension());
        GradientDescentValues gdv = lr.gradientDescent(X, y, theta, alpha, 
                num_iters);

        System.out.println("Theta computed from gradient descent:");
        gdv.getTheta().print(3,3);

        Matrix _X = new Matrix(3,1);
        _X.set(0,0,1);
        _X.set(1,0,((45-fnv.getMu().get(0, 0))/fnv.getSigma().get(0, 0)));
        _X.set(2,0,((85-fnv.getMu().get(0, 1))/fnv.getSigma().get(0, 1)));

        Matrix prob = lr.sigmoid(gdv.getTheta().transpose().times(_X));

        System.out.print("For a student with scores 45 and 85, we predict an"
                + " admission probability of ");
        prob.print(3, 3);

        Matrix p = lr.predict(gdv.getTheta(), X);

        System.out.println("Train accuracy: " + lr.accuracy(p, y));
        
       

    }	
}
