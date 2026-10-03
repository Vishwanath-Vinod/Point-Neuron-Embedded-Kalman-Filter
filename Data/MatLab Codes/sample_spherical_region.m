function points = sample_spherical_region(center, radius, num_points)
%SAMPLE_SPHERICAL_REGION Uniformly samples points within a spherical region 
% with optional z-plane constraints.
%
% Inputs:
%   center      - [x, y, z] coordinates of sphere center
%   radius      - Radius of the sphere
%   z_min       - Minimum z-value 
%   z_max       - Maximum z-value
%   num_points  - Number of points to sample
%
% Output:
%   points      - num_points x 3 matrix of sampled [x, y, z] coordinates

    points = zeros(num_points, 3);
    count = 0;
    while count < num_points
        u = rand();                       % radius
        theta = 2*pi*rand();              % azimuth
        phi = acos(2*rand() - 1);       % polar angle
        r_s = radius * u^(1/3);           % uniform in volume

        % Convert to Cartesian
        x = r_s * sin(phi) * cos(theta);
        y = r_s * sin(phi) * sin(theta);
        z = r_s * cos(phi);
        pt = center + [x, y, z];

        count = count + 1;
        points(count, :) = pt;
    end
end
