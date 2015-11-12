
import Jama.Matrix;
import MultivariateLinearRegression.FeatureNormalizationValues;
import MultivariateLinearRegression.GradientDescentValues;
import MultivariateLinearRegression.MultivariateLR;
import java.io.FileNotFoundException;
import java.io.IOException;
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
    public GradientDescentValues gradientDescent(Matrix X, Matrix y, Matrix theta, double alpha, int numIterations){
        //Write equivalent Java code for the Octave code below.
        
        //Initialize some useful values.
        //Octave: m = length(y); % number of training examples

        //create a matrix that stores cost history
        //Octave: J_history = zeros(num_iters, 1);

        //Loop thru numIterations
        //Octave:for iter = 1:num_iters
        
        // ====================== YOUR CODE HERE ======================
        // Instructions: Perform a single gradient step on the parameter vector
        //               theta. 
        //
        // Hint: While debugging, it can be useful to print out the values
        //       of the cost function (computeCostMulti) and gradient here.
        //

        
        
        
        
        
        
        
        
        
        // Save the cost J in every iteration    
        //Octave: J_history(iter) = costFunction(theta, X, y);
        return null;
    }

    /**
		%COSTFUNCTION Compute cost and gradient for logistic regression
		%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
		%   parameter for logistic regression and the gradient of the cost
		%   w.r.t. to the parameters.
	*/
	double costFunctionGD(Matrix theta, Matrix X, Matrix y){
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
			%
			% Note: grad should have the same dimensions as theta
			%
			% h = compute hypothesis
			% J = cost function
			%
		*/


	}


	/**
		%COSTFUNCTION Compute cost and gradient for logistic regression
		%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
		%   parameter for logistic regression and the gradient of the cost
		%   w.r.t. to the parameters.
	*/
	CostFunctionValues costFunction(Matrix theta, Matrix X, Matrix y){
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
			%
			% Note: grad should have the same dimensions as theta
			%
			% h = compute hypothesis
			% J = cost function
			% grad = gradient
			%
		*/
            int m = y.getRowDimension();
            double J = 0;
            int row = theta.getRowDimension();
            int col = theta.getColumnDimension();
            
            Matrix grad = new Matrix(row, col);
            double c = 1.d/m;
            
            Matrix hyp = sigmoid(X.times(theta));
            Matrix one = Util.ones(m, 1);
            
            J = y.uminus().times(log(hyp)).minus(one.minus(y).arrayTimes(one.minus(hyp))).get(0, 0);
            return null;
            

	}
        
        Matrix log(Matrix hyp){
            for(int r=0; r<hyp.getRowDimension(); r++){
                hyp.set(r, 0, Math.log(hyp.get(r, 0)));
            }
            
            return hyp;
        }

	/*
	function g = sigmoid(z)
	%SIGMOID Compute sigmoid function
	%   J = SIGMOID(z) computes the sigmoid of z.
	*/
	Matrix sigmoid(Matrix z){
		/*
		% You need to return the following variables correctly 
		g = zeros(size(z));

		% ====================== YOUR CODE HERE ======================
		% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
		%               vector or scalar).

		g = 1./(1+exp(-z));
		% ============================================================
		*/
            int row = z.getRowDimension();
            int col = z.getColumnDimension();
            Matrix g = new Matrix(row, col);
            
            for(int r=0; r<row; r++){
                g.set(r, 0, (1/(1+Math.exp(z.get(r, 0)))));
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
		/*
		m = size(X, 1); % Number of training examples

		% You need to return the following variables correctly
		p = zeros(m, 1);

		% ====================== YOUR CODE HERE ======================
		% Instructions: Complete the following code to make predictions using
		%               your learned logistic regression parameters. 
		%               You should set p to a vector of 0's and 1's
		%
		for i=1:m
		  h = sigmoid(X(i,:)*theta)
		  if h >= 0.5
		     p(i) = 1;
		  else
		     p(i) = 0;
		  end
		end
		% =========================================================================
		*/






	}


	/**
		fprintf('Train Accuracy: %f\n', );		
	*/
	double accuracy(Matrix p, Matrix y){
		//mean(double(p == y)) * 100



	}



	public static void main(String[] args) throws FileNotFoundException, IOException{
            
            LogReg lr = new LogReg();
            
		/**
			%% Load Data
			%  The first two columns contains the exam scores and the third column
			%  contains the label.

			data = load('ex2data1.txt');
			X = data(:, [1, 2]); y = data(:, 3);
		*/
		//===== JAVA CODE HERE ====
            
            Matrix X = Util.data("ex2data1.txt", 1, 2);
            Matrix y = Util.data("ex2data1.txt", 3);		

		/*
		[X mu sigma] = featureNormalize(X);
		*/
		//===== JAVA CODE HERE ====
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




		/**

			Compute Cost and Gradient
			%  Setup the data matrix appropriately, and add ones for the intercept term
			[m, n] = size(X);

			% Add intercept term to x and X_test
			X = [ones(m, 1) X];

			% Initialize fitting parameters
			initial_theta = zeros(n + 1, 1);

			% Compute and display initial cost and gradient
			[cost, grad] = costFunction(initial_theta, X, y);

			fprintf('Cost at initial theta (zeros): %f\n', cost);
			fprintf('Gradient at initial theta (zeros): \n');
			fprintf(' %f \n', grad);
			cost =  0.20350
			theta =

			  -25.16127
			    0.20623
			    0.20147


			

		*/
		//===== JAVA CODE HERE ====
            X = Util.insertX0(X);
            
            Matrix initial_theta = new Matrix(X.getColumnDimension(), 1);
            
            CostFunctionValues cfv = lr.costFunction(initial_theta, X, y);

            System.out.println("Cost at initial theta (zeroes): " + cfv.getJ());
            System.out.println("Gradient at initial theta (zeroes):");
            System.out.println(cfv.getGrad());

            return;


		/**
			Optimizing using GD
	        % Choose some alpha value
	        alpha = 0.1;
	        num_iters = 1000;

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
		*/

	    //===== JAVA CODE HERE ====

	        



	    /**
	    Predict and Accuracies
	    prob = sigmoid([1 45 85] * theta);
		fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
		         'probability of %f\n\n'], prob);

		% Compute accuracy on our training set
		p = predict(theta, X);

	    */
		//===== JAVA CODE HERE ====


		/**
		fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
		*/
		//===== JAVA CODE HERE ====




	}	
}
