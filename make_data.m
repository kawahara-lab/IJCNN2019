%This file processes downloaded data for experiment in paper
%put output of this file in ./data

%% labels
labels = string;
labels(1,1:60) = 'sitting';
labels(1,60+1:60*2) = 'standing';
labels(1,60*2+1:60*3) = 'lying on back';
labels(1,60*3+1:60*4) = 'lying on right side';
labels(1,60*4+1:60*5) = 'ascending stairs';
labels(1,60*5+1:60*6) = 'descending stairs';
labels(1,60*6+1:60*7) = 'standing in an elevator';
labels(1,60*7+1:60*8) = 'moving in an elevator';
labels(1,60*8+1:60*9) = 'walking';
labels(1,60*9+1:60*10) = 'walking on a treadmill flat';
labels(1,60*10+1:60*11) = 'walking on a treadmill 15deg';
labels(1,60*11+1:60*12) = 'running on a treadmill';
labels(1,60*12+1:60*13) = 'exercising on a stepper';
labels(1,60*13+1:60*14) = 'exercising on a cross trainer';
labels(1,60*14+1:60*15) = 'cycling in horizontal positions';
labels(1,60*15+1:60*16) = 'cycling in vertical positions';
labels(1,60*16+1:60*17) = 'rowing';
labels(1,60*17+1:60*18) = 'jumping';
labels(1,60*18+1:60*19) = 'playing basketball';

%% load files

label = {};
motiondata = {};
for i = 1:8
    person_number = strcat('p', num2str(i));
    tmp = {};
    for n = 1:19
        if n <= 9
            dir_name = strcat('a0', num2str(n));
        else
            dir_name = strcat('a', num2str(n));
        end
        
        cd(dir_name)
        cd(person_number)
      
        files = dir('*.txt');
        X = cell(1,length(files));
        
        for k = 1:length(files)
            X{k} = (dlmread(files().name,',')).';
        end
        tmp = horzcat(tmp,X);
        motiondata = horzcat(motiondata,X);
        cd ../..
    end
    eval(['motiondata' num2str(i) '= tmp;']);
    eval(['labels' num2str(i) '= labels;']);
    label = horzcat(label,labels);
end

save('DSADS.mat', 'motiondata1', 'labels')
clear dir_name files i k n person_number tmp X
save('DSADS_all.mat')
