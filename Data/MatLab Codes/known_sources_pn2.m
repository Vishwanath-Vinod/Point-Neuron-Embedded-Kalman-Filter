function [bias, theta_info] = known_source_pns1(sources, MicCoord)
    % Parameters
    Lx = 1.3;
    Ly = 1.6;
    min_dist = 0.05;  % Minimum distance from microphones
    target_center = [0.65, 0.80, 0.525];
    target_radius = 0.1;

    num_circle_pts = 1;
    num_wall_pts = 8;

    bias = [];
    theta_info = [];
    global_idx = 0;
    
    % --- 1. Add circular PNs for each source ---
    for s = 1:size(sources, 1)
        src = sources(s, :);
        src_z = src(3);
    
        % Circle around the source
        radius = 0;
        theta = linspace(0, 2*pi, num_circle_pts + 1); theta(end) = [];
        circ_x = src(1) + radius * cos(theta);
        circ_y = src(2) + radius * sin(theta);
        circ_z = src_z * ones(size(circ_x));
        circ_pts = [circ_x', circ_y', circ_z'];
    
        % --- Filtering circular points ---
        dist_to_target = vecnorm(circ_pts - target_center, 2, 2);
        keep = dist_to_target > 0;
    
        if ~isempty(MicCoord)
            for m = 1:size(MicCoord, 1)
                dist_to_mic = vecnorm(circ_pts - MicCoord(m, :), 2, 2);
                keep = keep & (dist_to_mic > min_dist);
            end
        end
    
        circ_pts = circ_pts(keep, :);
        kept_theta = theta(keep);
    
        % Save theta info and add to bias
        for i = 1:size(circ_pts, 1)
            global_idx = global_idx + 1;
            theta_info = [theta_info; global_idx, kept_theta(i), s];
        end
    
        bias = [bias; circ_pts];
        disp(['After source ', num2str(s), ', bias size: ', mat2str(size(bias))]);
    end
    
    % --- 2. Add wall PNs only once ---
    wall_pts = [];
    z_levels = [0.525];
    
    for zi = 1:length(z_levels)
        current_z = z_levels(zi);
        wall_z = current_z * ones(num_wall_pts, 1);
    
        % Walls (x = 0 or x = Lx)
        wall_y = linspace(0, Ly, ceil(num_wall_pts / 2))';
        wall_pts = [wall_pts; ...
            zeros(size(wall_y)), wall_y, wall_z(1:length(wall_y)); ...
            Lx * ones(size(wall_y)), wall_y, wall_z(1:length(wall_y))];
    
        % Walls (y = 0 or y = Ly)
        wall_x = linspace(0, Lx, floor(num_wall_pts / 2))';
        wall_pts = [wall_pts; ...
            wall_x, zeros(size(wall_x)), wall_z(1:length(wall_x)); ...
            wall_x, Ly * ones(size(wall_x)), wall_z(1:length(wall_x))];
    end
    
    % --- Final filtering for wall PNs ---
    dist_to_target = vecnorm(wall_pts - target_center, 2, 2);
    keep = dist_to_target > target_radius;
    
    if ~isempty(MicCoord)
        for m = 1:size(MicCoord, 1)
            dist_to_mic = vecnorm(wall_pts - MicCoord(m, :), 2, 2);
            keep = keep & (dist_to_mic > min_dist);
        end
    end
    
    wall_pts = wall_pts(keep, :);
    bias = [bias; wall_pts];
    
    % Update global indices (no theta info for wall PNs)
    for i = 1:size(wall_pts, 1)
        global_idx = global_idx + 1;
    end
    
    disp(['Final bias size (circular + wall): ', mat2str(size(bias))]);