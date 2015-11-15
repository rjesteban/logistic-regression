
import Jama.Matrix;
import MultivariateLinearRegression.GradientDescentValues;
import java.io.IOException;
import util.DataPlot;
import util.Util;

public class RegularizedLogReg extends LogReg{


	/**
		%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
		%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
		%   theta as the parameter for regularized logistic regression and the
		%   gradient of the cost w.r.t. to the parameters. 
	*/

    CostFunctionValues costFunctionReg(Matrix theta, Matrix X, Matrix y, double lambda){
        /**
                % Initialize some useful values
                m = length(y); % number of training examples

                % You need to return the following variables correctly 
                J = 0;
                grad = zeros(size(theta));

                % ====================== YOUR CODE HERE ======================
                % Instructions: Compute the cost of a particular choice of theta.
                %               You should set J to the cost.
                %               Compute the partial derivatives and set grad to the partial
                %               derivatives of the cost w.r.t. each parameter in theta
        */
        int m = y.getRowDimension();
        CostFunctionValues cfv = costFunction(theta, X, y);
        
        //============= add regularization term to cost ===============
        
        Matrix thet = theta.getMatrix(1, theta.getRowDimension()-1, 0, 0);
        thet = pow(thet, 2).times(lambda/2*m);
        Matrix k = new Matrix(1,1);
        
        for(int r=0; r<thet.getRowDimension(); r++){
            k.set(0,0,k.get(0,0)+thet.get(r, 0));
        }

        cfv.setJ(cfv.getJ().plus(k));
        
        //============= gradient: add regularization term ===============
        
        thet = theta.getMatrix(1, theta.getRowDimension()-1, 0, 0);
        thet.timesEquals(lambda/m);
        
        Matrix grad = cfv.getGrad();
        
        for(int r=1; r< cfv.getGrad().getRowDimension(); r++){
            grad.set(r, 0, grad.get(r,0)+thet.get(r-1, 0));
        }
        cfv.setGrad(grad);
        return cfv;
    }
    
    public GradientDescentValues gradientDescent(Matrix X, Matrix y, 
            Matrix theta, double alpha, int numIterations, double lambda){
        int m = y.getRowDimension();
        Matrix J_history = new Matrix(numIterations,1);
        
        
        
        for(int iter=0;iter<numIterations;iter++){
            CostFunctionValues cfv = costFunctionReg(theta, X, y, lambda);
            J_history.set(iter, 0, cfv.getJ().get(0, 0));
            theta = theta.minus(cfv.getGrad().times(alpha));
        }
        
        return new GradientDescentValues(theta, J_history);
    }
        
        
        

    /**
            % MAPFEATURE Feature mapping function to polynomial features
            %
            %   MAPFEATURE(X1, X2) maps the two input features
            %   to quadratic features used in the regularization exercise.
            %
            %   Returns a new feature array with more features, comprising of 
            %   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
            %
            %   Inputs X1, X2 must be the same size
            %
    */
    Matrix mapFeature(Matrix X1, Matrix X2){
        int degree = 6;
        Matrix out = Util.ones(X1.getRowDimension(), X1.getColumnDimension());

        for(int i=1; i<=degree; i++){
            for(int j=0; j<=i; j++){
                out = Util.append(out, pow(X1,(i-j)).arrayTimes(pow(X2,j)));
            }
        }
        return out;
    }

    Matrix pow(Matrix X, int pow){
        Matrix N = X.copy();

        for(int r=0;r<N.getRowDimension();r++){
            N.set(r, 0, Math.pow(N.get(r, 0),pow));
        }
        return N;
    }
        
        
    public static void main(String[] args) throws IOException{

        RegularizedLogReg rlr = new RegularizedLogReg();

        /**
            %% Load Data
            %  The first two columns contains the X values and the third column
            %  contains the label (y).

            data = load('ex2data2.txt');
            X = data(:, [1, 2]); y = data(:, 3);

            plotData(X, y);

            % Put some labels 
            hold on;

            % Labels and Legend
            xlabel('Microchip Test 1')
            ylabel('Microchip Test 2')

            % Specified in plot order
            legend('y = 1', 'y = 0')
        */
        //========= JAVA CODE =============
        Matrix X = Util.data("ex2data2.txt", 1,2);
        Matrix y = Util.data("ex2data2.txt", 3);
        
        new Thread(new DataPlot(X, y, "Microchip Test 1", "Microchip Test 2")).start();
        
        //put some labels

        /**
            % Add Polynomial Features

            % Note that mapFeature also adds a column of ones for us, so the intercept
            % term is handled
            X = mapFeature(X(:,1), X(:,2));
        */
        //========= JAVA CODE =============
        X = rlr.mapFeature(X.getMatrix(0, X.getRowDimension()-1, 0, 0),
                        X.getMatrix(0, X.getRowDimension()-1, 1, 1));
        /**

            % Initialize fitting parameters
            initial_theta = zeros(size(X, 2), 1);

            % Set regularization parameter lambda to 1
            lambda = 1;

        */	
        Matrix initial_theta = new Matrix(X.getColumnDimension(), 1);
        double lambda = 1;
        /**
            Optimizing using GD
            % Choose some alpha value
            alpha = 0.01;
            num_iters = 400;

            % Init Theta and Run Gradient Descent 
            theta = zeros(3, 1);
            [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

            % Plot the convergence graph
            figure;
            plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
            xlabel('Number of iterations');
            ylabel('Cost J');

            % Display gradient descent's result
            fprintf('Theta computed from gradient descent: \n');
            fprintf(' %f \n', theta);
        */

        //===== JAVA CODE HERE ====

            double alpha = 1;
            int num_iters = 1000;
            System.out.println("cost: ");
            rlr.costFunctionReg(initial_theta, X, y, lambda).getJ().print(3, 6);
            
            GradientDescentValues gdv = rlr.gradientDescent(X, y, initial_theta, alpha, num_iters, lambda);
            
            System.out.println("Theta computed from gradient descent:");
            gdv.getTheta().print(6,3);
            
            Matrix p = rlr.predict(gdv.getTheta(), X);
            System.out.println("Train Accuracy: "+ rlr.accuracy(p, y));
            

    }

}