
import Jama.Matrix;

class RegularizedLogReg extends LogReg{


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
            /**
                    degree = 6;
                    out = ones(size(X1(:,1)));
                    for i = 1:degree
                        for j = 0:i
                            out(:, end+1) = (X1.^(i-j)).*(X2.^j);
                        end
                    end
            */

    }
    public static void main(String[] args){
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




            /**
                    % Add Polynomial Features

                    % Note that mapFeature also adds a column of ones for us, so the intercept
                    % term is handled
                    X = mapFeature(X(:,1), X(:,2));
            */
            //========= JAVA CODE =============



            /**

                    % Initialize fitting parameters
                    initial_theta = zeros(size(X, 2), 1);

                    % Set regularization parameter lambda to 1
                    lambda = 1;

            */	


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