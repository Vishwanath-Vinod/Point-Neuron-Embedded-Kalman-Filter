clear 
rng('default');
rng(1);
L               = [1.3,1.6,1.05]; % Room dimensions
beta            = [0.8,0.8,0.8,0.8,0,0]; % Reflection coefficients
Frequency       = 900; % Frequency
c               = 343; % Speed of sound
k               = 2*pi*Frequency/c; % Wave Number

% Target region center and radius
target_center   = [0.65,0.8,0.525];
target_radius   = 0.50;


num_sources = 1;
sources = zeros(num_sources, 3);  % Preallocate
count = 0;
sources(1,:) = [0.2,0.2,0.525];

disp(sources);

% Number of microphones based on frequency Q=2*Ns+1
target_radius = min(L(1), L(2))/2 - 0.05;
Ns              = k*target_radius;
num_mics        = ((2*ceil(Ns)+1)) ;

% Allot microphones randomly within the spherical target region and take readings
MicCoord = arrange_mics_on_boundary(L,0.525,num_mics,0.1); %sample_spherical_region(target_center, target_radius, num_mics);
MicData = zeros(size(MicCoord, 1), 1);  
for i = 1:size(sources, 1)
    MicData = MicData + ism(k, MicCoord, sources(i, :), L, beta, 10);
end
MicData = awgn((MicData),20,'measured');

% Number of Point Neurons
V = 17;
% Initialize weights
angles = 2*pi*rand(V,1);
weights = (2*rand(V,1)-1) .* exp(1j * angles);
% Initialize PN locations (biases)
biases = [known_source_pns1(sources, MicCoord)];
disp(size(biases))

%Sample pressure readings in the target region
num_samples_target = 1500;
sample_center = [0.65,0.8,0.525];
sampling_target = sample_circular_region(sample_center, target_radius, num_samples_target);
pressure_target = zeros(size(sampling_target, 1), 1);  
for i = 1:size(sources, 1)
    pressure_target = pressure_target + ism(k, sampling_target, sources(i, :), L, beta, 10);
end

%Sample pressure readings in the whole room

% Spatial resolution
dx = 0.025;
dy = 0.025;
dz = 0;

% Define grid points along each axis
x = 0:dx:L(1);
y = 0:dy:L(2);
z = 0.525;
% Generate 3D meshgrid
[X, Y, Z] = meshgrid(x, y, z);
sampling_room = [X(:), Y(:), Z(:)];
pressure_room = zeros(size(sampling_room, 1), 1);  
for i = 1:size(sources, 1)
    pressure_room = pressure_room + ism(k, sampling_room, sources(i, :), L, beta, 10);
end

% Save mic positions and mic signals
save('MIC_DATA_ms.mat', 'MicCoord', 'MicData');
save('P_TARGET_ms.mat', 'sampling_target', 'pressure_target');
save('P_ROOM_ms.mat', 'sampling_room', 'pressure_room');
save('PN_INITIALIZATION_ms.mat', 'weights', 'biases');

% Parameters for motion
T_total = 100;                     % Total time in seconds
dt = 1;                            % Time step (s)
num_steps = round(T_total / dt);
v = [0.004, 0.003, 0];                 % Velocity vector in m/s

% Initialize trajectory
trajectory = zeros(num_steps, 3);
accel_std_true = [0.0001, 0.0001, 0];
accel_samples = zeros(num_steps-1, 3);
for i = 1:num_steps
    t = (i - 1) * dt;
    trajectory(i, :) = sources(1,:) + v * t;
    
    % Clip to room boundaries
    trajectory(i, :) = max(trajectory(i, :), [0 0 0] + 0.05); 
    trajectory(i, :) = min(trajectory(i, :), L - 0.05);
    if i < num_steps
        acc_noise = accel_std_true .* randn(1, 3);  % random accel
        v = v + acc_noise * dt;
        accel_samples(i, :) = acc_noise;
    end
end
% Save trajectory to CSV
writematrix(trajectory, 'trajectory-1.csv');
MicData_time = zeros(num_mics, num_steps);  % Each column is data at time step i
MicData_time_noisy = zeros(num_mics, num_steps);
for t = 1:num_steps
    src_pos = trajectory(t, :);
    MicData_time(:, t) = ism(k, MicCoord, src_pos, L, beta, 10);  % ISM at each step
    MicData_time_noisy(:, t) = awgn(MicData_time(:, t), 20, 'measured');
end

% Estimate noise vector
noise = MicData_time_noisy - MicData_time;
% Averaging across all mics and time
sigma2 = mean(var([real(noise); imag(noise)], 0, 2));
R = 1e-2 * eye(2 * num_mics);
disp(sigma2)
% Estimate standard deviation of acceleration
accel_std_estimated = std(accel_samples, 0, 1);  % [std_x, std_y, std_z]
disp(accel_std_estimated)
save('Process_noise.mat', 'accel_std_estimated');
save('Measurements.mat','MicData_time_noisy')
save('Measurement_noise.mat', 'R');

figure;
hold on;

% --- Plot the room as a transparent cube ---
room_origin = [0, 0, 0];         % Lower corner
room_size   = L;                 % Dimensions
room_color  = [0.8 0.8 1];       % Light blue
alpha_val   = 0.05;              % Transparency

estimated_trajectory1 = readmatrix('est_trajectory.csv');
trajectory = readmatrix('trajectory.csv');
plotcube(room_size, room_origin, alpha_val, room_color);
hold on;  % Ensure all plots go on the same figure

% --- Plot the point neurons ---
h1 = plot3(biases(:,1), biases(:,2), biases(:,3), 'x', 'Color', [0.5 0.5 0.5], ...
    'LineWidth', 1.5);

% --- Plot the source trajectory ---
h2 = plot3(trajectory(:,1), trajectory(:,2), trajectory(:,3), 'r.-', ...
    'LineWidth', 1.5);

% --- Plot the estimated trajectory ---
h3 = plot3(estimated_trajectory1(:,1), estimated_trajectory1(:,2), estimated_trajectory1(:,3), ...
    'g.-', 'LineWidth', 1.5);

% --- Plot the microphones ---
h4 = plot3(MicCoord(:,1), MicCoord(:,2), MicCoord(:,3), 'b.', ...
    'MarkerSize', 5);

xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Source Tracking');
axis equal;
grid on;
xlim([0 L(1)]);
ylim([0 L(2)]);
zlim([0 L(3)]);
view(3);

% --- Add legend explicitly using handles ---
legend([h1, h2, h3, h4], ...
       {'Point Neurons', 'Source Trajectory', 'Estimated Trajectory', 'Microphones'}, ...
       'Location', 'bestoutside');


% --- Save figure ---
saveas(gcf, 'mic_source_pn_plot_ms.png');

