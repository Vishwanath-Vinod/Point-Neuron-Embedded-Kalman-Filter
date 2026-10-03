classdef HENGLIB
% Some useful functions for acoustic research by Shaoheng Xu
% ======================================================================= %
% File            : HENGLIB.m
% Creation        : 31-MAY-2023
% Author          : Shaoheng Xu
% Last Modified   : 28-Jun-2023
% ----------------------------------------------------------------------- &
% 
% Description: 
%       Some useful functions I used for acoustic research in my work.
% 
% Instructions: 
%       1) Add the HENGLIB.m to the MATLAB working-folder (or path)
%       2) Call function by: output = HENGLIB."function_name"(Inputs)
% 
% Notes and Assumptions:
%       Unless stated otherwise, these functions follow the notation from
%       the Fourier Acoustic book.
% ======================================================================= %

% ======================================================================= %
%% - P R O P E R T I E S -
% ======================================================================= %
methods(Static)

% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %

function [x, ompErr] = complexOMP(y, Psi, sparsity, thres)
% ----------------------------------------------------------------------- %
% [x, ompErr] = complexOMP(y, Psi, k, thres)
% ----------------------------------------------------------------------- %
% Creation       : 31-May-2023
% Last Modified  : 28-Jun-2023
% Author         : Shaoheng Xu
% ----------------------------------------------------------------------- %
% Description: 
%       The function uses orthogonal matching pursuit (OMP) to solve the
%       sparse approximation problem over redundant dictionaries for
%       complex cases (i.e., complex measurement vector, complex dictionary
%       and complex additive white Gaussian noise).
% 
%       Note that the sparsity of the output "x" is less than or equal to 
%       the desired input sparsity (i.e., |x|_0 <= k).
% ----------------------------------------------------------------------- %
% Input: 
%       1) y        - M x 1  - the measurement vector
%       2) Psi      - M x N  - the dictionary
%       3) sparsity - scaler - the number of iterations, also the sparsity 
%                              of the solution
%       4) thres    - scaler - (optional) the error threshold
% 
% Output:
%       1) x      - N x 1        - the sparse solution of the 
%                                  approximation problem
%       2) ompErr - 1 x sparsity - the l2-norm of the residual after
%                                  each iteration
% ----------------------------------------------------------------------- %

% Choose a default value for thres.
if nargin < 4
    thres = 0;
end

% Step 1: Initialize the residual r0 = y and the set of selected atom
% c = φ. Let iteration counter i = 1.

resid = y;
c = [];

% Create an empty vector to store the l2-norm of residual after each
% iteration.
ompErr = NaN(1, sparsity);

for i = 1 : sparsity

    % Step 2: Find the variable psi_t that solves the maximization problem
    % arg max(t) |psi_t^H * r_(i-1)|. Add the variable to the set of
    % selected variables. Update c(i) = c(i-1) U {psi_t}.
    ti = HENGLIB.cmpxMutlIncoh(Psi, resid);
    c = union(c, ti);

    % Step 3: Let Pi denote the projection onto the linear space spanned by
    % the elements of Psi(c(i)).
    Psi_ci = Psi(:, c);
    Pi = Psi_ci / (ctranspose(Psi_ci) * Psi_ci) * ctranspose(Psi_ci);
    % Update the residual ri.
    resid = (eye(size(Pi)) - Pi) * y;

    % Step 4: If the stopping condition is achieved (i.e., the l2-norm of
    % ri < thres), go to Step 5. Otherwise, set i = i + 1 and go back to
    % Step 2 until reaching the given threshold or maximum iterative times.
    ompErr(i) = norm(resid);
    if ompErr(i) < thres
        break;
    end
end

% Step 5: Calculate the vector x with LS method.
Psi(:, setdiff(1:size(Psi,2), c)) = 0;
% Method 1.
% x = lsqr(Psi, y);
% Method 2.
x = pinv(Psi) * y;

% Remove the deactivated weights from the output "x".
x(setdiff(1:numel(x), c)) = 0;

end

% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %

function [x, ompErr] = complexOMP2(y, Psi, sparsity, thres)
% ----------------------------------------------------------------------- %
% [x, ompErr] = complexOMP2(y, Psi, sparsity, thres)
% ----------------------------------------------------------------------- %
% Creation       : 28-Jun-2023
% Last Modified  : 28-Jun-2023
% Author         : Shaoheng Xu
% ----------------------------------------------------------------------- %
% Description: 
%       The function uses orthogonal matching pursuit (OMP) to solve the
%       sparse approximation problem over redundant dictionaries for
%       complex cases (i.e., complex measurement vector, complex dictionary
%       and complex additive white Gaussian noise).
% 
%       Note that the sparsity of the output "x" equals to the desired
%       input sparsity (i.e., |x|_0 = k).
% ----------------------------------------------------------------------- %
% Input: 
%       1) y        - M x 1  - the measurement vector
%       2) Psi      - M x N  - the dictionary
%       3) sparsity - scaler - the number of iterations, also the sparsity 
%                              of the solution
%       4) thres    - scaler - (optional) the error threshold
% 
% Output:
%       1) x      - N x 1        - the sparse solution of the 
%                                  approximation problem
%       2) ompErr - 1 x sparsity - the l2-norm of the residual after
%                                  each iteration
% ----------------------------------------------------------------------- %

% Choose a default value for thres.
if nargin < 4
    thres = 0;
end

% Step 1: Initialize the residual r0 = y and the set of selected atom
% c = φ. Let iteration counter i = 1.

resid = y;
c = [];
Psi_edt = Psi;

% Create an empty vector to store the l2-norm of residual after each
% iteration.
ompErr = NaN(1, sparsity);

for i = 1 : sparsity

    % Step 2: Find the variable psi_t that solves the maximization problem
    % arg max(t) |psi_t^H * r_(i-1)|. Add the variable to the set of
    % selected variables. Update c(i) = c(i-1) U {psi_t}.
    ti = HENGLIB.cmpxMutlIncoh(Psi_edt, resid);
    c = union(c, ti);

    % Step 3: Let Pi denote the projection onto the linear space spanned by
    % the elements of Psi(c(i)).
    Psi_ci = Psi(:, c);
    Pi = Psi_ci / (ctranspose(Psi_ci) * Psi_ci) * ctranspose(Psi_ci);
    % Update the residual ri.
    resid = (eye(size(Pi)) - Pi) * y;
    % Update the edited dictionary Psi_edt.
    Psi_edt(:, c) = 0;

    % Step 4: If the stopping condition is achieved (i.e., the l2-norm of
    % ri < thres), go to Step 5. Otherwise, set i = i + 1 and go back to
    % Step 2 until reaching the given threshold or maximum iterative times.
    ompErr(i) = norm(resid);
    if ompErr(i) < thres
        break;
    end
end

% Step 5: Calculate the vector x with LS method.
Psi(:, setdiff(1:size(Psi,2), c)) = 0;
% Method 1.
% x = lsqr(Psi, y);
% Method 2.
x = pinv(Psi) * y;

% Remove the deactivated weights from the output "x".
x(setdiff(1:numel(x), c)) = 0;

end

% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %

function [ti] = cmpxMutlIncoh(Psi, Res)
% ----------------------------------------------------------------------- %
% [ti] = cmpxMutlIncoh(Psi, Res)
% ----------------------------------------------------------------------- %
% Creation       : 31-May-2023
% Last Modified  : 31-May-2023
% Author         : Shaoheng Xu
% ----------------------------------------------------------------------- %
% Description: 
%       The function compute the mutual incoherence between the dictionary
%       matrix (Psi) and the residual (Res). Then it returns the
%       corresponding index number of the selected variable.
% ----------------------------------------------------------------------- %
% Input: 
%       1) Psi - the dictionary
%       2) Res - the current residual
% 
% Output:
%       1) ti - the index number of the selected variable (column)
% ----------------------------------------------------------------------- %

% Conduct column-normalization for the matrix "Psi" and the residual vector
% "Res".
[Psi] = HENGLIB.colNorm(Psi);
[Res] = HENGLIB.colNorm(Res);

% The number of variables in the matrix Psi, i.e., the number of column in
% the matrix Psi.
noOfVar = size(Psi, 2);

% The incoherence of each variable of Psi with the residual.
incoh = NaN(1, noOfVar);
for ic = 1 : noOfVar
    incoh(ic) = abs(ctranspose(Psi(:, ic)) * Res);
end

% Find the mutual incoherence and its corresponding index number.
[~, ti] = max(incoh);

end

% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %

function [Matrix_Normd] = colNorm(Matrix)
% ----------------------------------------------------------------------- %
% [Matrix_Normd] = colNorm(Matrix)
% ----------------------------------------------------------------------- %
% Creation       : 01-Jun-2023
% Last Modified  : 01-Jun-2023
% Author         : Shaoheng Xu
% ----------------------------------------------------------------------- %
% Description: 
%       The function normalizes each column in the matrix and then returns
%       the normalized result.
% ----------------------------------------------------------------------- %
% Input: 
%       1) Matrix - the original matrix
% 
% Output:
%       1) Matrix_Normd - the output matrix with normalized columns.
% ----------------------------------------------------------------------- %

% Create an empty matrix to store the results.
Matrix_Normd = NaN(size(Matrix));

% Normalize each column in the original matrix.
for ic = 1 : size(Matrix,2)
    Matrix_Normd(:, ic) = Matrix(:, ic) / norm(Matrix(:, ic));
end

end

% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %

function [xCoord, yCoord, zCoord] = getMeshgrid(centre, sideLen, sidePts)
% ----------------------------------------------------------------------- %
% [xCoord, yCoord, zCoord] = getMeshgrid(centre, sideLen, sidePts)
% ----------------------------------------------------------------------- %
% Creation       : 08-Jun-2023
% Last Modified  : 08-Jun-2023
% Author         : Shaoheng Xu
% ----------------------------------------------------------------------- %
% Description: 
%       The function generates a mesh (rectangular) grid of points that are
%       located on the same horizontal plane.
% ----------------------------------------------------------------------- %
% Input: 
%       1) centre - the centre of the rectangular grid (scalar or 1-by-3 or
%                   3-by-1)
%       2) sideLen - the length of the rectangular grid (scalar or 2-by-1  
%                    or 1-by-2)
%       3) sidePts - the number of points along one side of the grid (
%                    scalar or 2-by-1 or 1-by-2)
% 
% Output:
%       1) xCoord - a list of x-coordinates of the grid points (N-by-1)
%       2) yCoord - a list of y-coordinates of the grid points (N-by-1)
%       3) zCoord - a list of z-coordinates of the grid points (N-by-1)
% ----------------------------------------------------------------------- %

if isscalar(centre)
    xCen = centre;
    yCen = centre;
    zCen = centre;
elseif numel(centre) == 3
    xCen = centre(1);
    yCen = centre(2);
    zCen = centre(3);
else
    error("Please input a correct size for the variable 'centre'.");
end

% "sx" and "sy" define the possible range of the rectangular grid.
if isscalar(sideLen)
    if isscalar(sidePts)
        sx = linspace(xCen-(sideLen/2), xCen+(sideLen/2), sidePts);
        sy = linspace(yCen-(sideLen/2), yCen+(sideLen/2), sidePts);
    else
        error('sidePoint should be a scaler.');
    end
elseif numel(sideLen) == 2
    if numel(sidePts) == 2
        lenX = sideLen(1);   lenY = sideLen(2);
        sx = linspace(xCen-(lenX/2), xCen+(lenX/2), sidePts(1));
        sy = linspace(yCen-(lenY/2), yCen+(lenY/2), sidePts(2));
    elseif isscalar(sidePts)
        lenX = sideLen(1);   lenY = sideLen(2);
        sx = linspace(xCen-(lenX/2), xCen+(lenX/2), sidePts);
        sy = linspace(yCen-(lenY/2), yCen+(lenY/2), sidePts);
    else
        error('The size of sidePoint is wrong.');
    end
end

% Determine the 3D-coordinate of each grid point.
[xCoord, yCoord] = meshgrid(sx, sy);
xCoord = reshape(xCoord, [], 1);
yCoord = reshape(yCoord, [], 1);
zCoord = zCen * ones(size(xCoord));

end

% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %

function [xCoord, yCoord, zCoord] = getCubegrid(centre, sideLen, sidePts)
% ----------------------------------------------------------------------- %
% [xCoord, yCoord, zCoord] = getCubegrid(centre, sideLen, sidePts)
% ----------------------------------------------------------------------- %
% Creation       : 08-Jun-2023
% Last Modified  : 08-Jun-2023
% Author         : Shaoheng Xu
% ----------------------------------------------------------------------- %
% Description: 
%       The function creates a cubic grid of points in the 3D space.
% ----------------------------------------------------------------------- %
% Input: 
%       1) centre - the centre of the cubic grid (scalar or 1-by-3 or 
%                   3-by-1)
%       2) sideLen - the side length of the cubic grid (scalar or 3-by-1  
%                    or 1-by-3)
%       3) sidePts - the number of points along one side of the grid (
%                    scalar or 3-by-1 or 1-by-3)
% 
% Output:
%       1) xCoord - a list of x-coordinates of the grid points (N-by-1)
%       2) yCoord - a list of y-coordinates of the grid points (N-by-1)
%       3) zCoord - a list of z-coordinates of the grid points (N-by-1)
% ----------------------------------------------------------------------- %

% Check the input variable: centre.
if isscalar(centre)
    xCen = centre;     yCen = centre;     zCen = centre;
elseif numel(centre) == 3
    xCen = centre(1);  yCen = centre(2);  zCen = centre(3);
else
    error("Please input a correct size for the variable 'centre'.");
end

% Check the input variable: sideLen.
if isscalar(sideLen)
    xLen = sideLen;     yLen = sideLen;     zLen = sideLen;
elseif numel(sideLen) == 3
    xLen = sideLen(1);  yLen = sideLen(2);  zLen = sideLen(3);
else
    error("Please input a correct size for the variable 'sideLen'.");
end

% Check the input variable: sidePts.
if isscalar(sidePts)
    xPts = sidePts;     yPts = sidePts;     zPts = sidePts;
elseif numel(sidePts) == 3
    xPts = sidePts(1);  yPts = sidePts(2);  zPts = sidePts(3);
else
    error("Please input a correct size for the variable 'sidePts'.");
end

% Create one layer of points (x-y plane) and record their x,y-coordinates.
[xCoord, yCoord, ~] = HENGLIB.getMeshgrid( ...
    [xCen,yCen,zCen], [xLen,yLen], [xPts,yPts]);
% Duplicate the x,y-coordinates for the whole cubic grid.
xCoord = repmat(xCoord, zPts, 1);
yCoord = repmat(yCoord, zPts, 1);

% Determine the z-coordinates of the cubic grid points.
zCoord = linspace(zCen-(zLen/2), zCen+(zLen/2), zPts);
zCoord = reshape(repmat(zCoord, xPts*yPts, 1), [], 1);

end

% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %

end

end