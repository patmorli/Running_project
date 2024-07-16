% create a matrix for all calculations I'd like to do 

clear; clc
resnet = ["frozen weights"; "unfrozen weights"; "no transfer"];
window_sizes = ["10k window";  "25k window"];
dropout = ["no dropout"];
spectrogram_nperseg = ["125"; "250"; "500"];
different_speeds = ["one speed", "all speeds"];


p = 1;
for speeds = 1:length(different_speeds)
    for window = 1:length(window_sizes)
        for spec = 1:length(spectrogram_nperseg)
            for res = 1:length(resnet)
                for drop = 1:length(dropout)
                
                    my_crazy_matrix(p) = strcat(window_sizes(window), ", ", spectrogram_nperseg(spec),  ", ", resnet(res), ", ", dropout(drop), ", ", different_speeds(speeds));
                    p = p + 1;
                end
            end
        end
    end

end
my_crazy_matrix = my_crazy_matrix';

writematrix(my_crazy_matrix, '/Volumes/GoogleDrive/My Drive/Running Plantiga Project/Data/Results/my_models.csv');
