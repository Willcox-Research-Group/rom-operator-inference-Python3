function a = save_as_tec(fname_raw)
a = 0;
global var_per_line var_per_line_grid fname_grid line_skip var_target var_cc var_node node_num element_num zone_type line_skip_grid input x_min x_max limit_value


mean_option=0; %0, simple extract the mean value  2, high pass filter

input=100; % 1, interp from plt file 2, CI volume 3, Experiment Data 100, Tecplot ASCII

var_per_line = 4; var_per_line_grid = 5; line_skip = 15; element_num = 38523; node_num = 39065; limit_value = [];
line_skip_grid = 9; zone_type = 'FEQuadrilateral'; var_cc = 1:8; var_node = []; x_min = []; x_max = []; variable = 'NonConsQv';

pod_option = 'mos'; time_step = 1; 

for t_end = 150000 

t_range = 159199 : time_step : 159199;

dimension_option=2; zone_num=1;

m_folder = ' ';


fname_grid = 'grid.dat';

var_select = 1:8; %var_cc; 
var_target = var_select;

var_name=['P    ';'U    ';'V    ';'T    ';'  CH4';'  O2 ';'  CO2';'  H2O'];

ndim = dimension_option;

read_format = '%f'; read_format_xyz = '%f'; prec = '%20.15E'; out_format = prec;
for j = 1 : var_per_line-1
    read_format = [read_format ' %f'];
    out_format = [out_format ' ' prec];
end
read_format = [read_format '\n'];
out_format = [out_format '\n'];

for j = 1 : var_per_line_grid-1
    read_format_xyz = [read_format_xyz ' %f'];
end
read_format_xyz = [read_format_xyz '\n'];

out_format_xyz = out_format;

read_format_c2n = '%f'; out_format_c2n = '%6d';
for j = 1 : (2^dimension_option-1)
    read_format_c2n = [read_format_c2n ' %f'];
    out_format_c2n = [out_format_c2n ' %6d'];
end
read_format_c2n = [read_format_c2n '\n'];
out_format_c2n = [out_format_c2n '\n'];

[xyz,c2n] = importGridFile(fname_grid,line_skip_grid,read_format_xyz,read_format_c2n,dimension_option,element_num,node_num);
        
num_c = element_num; var_num = length(var_select); It = length(t_range);

p = zeros(It,num_c*var_num);

mm = 0;
for i=t_range
    mm = mm + 1;

    file_name = [fname_raw '.dat'];
    out = importTecASCIIdata(file_name,line_skip,read_format,var_target,var_cc,var_node,node_num,element_num,c2n,ndim);

    for ivar = 1 : var_num
        p(mm,(1+(ivar-1)*num_c):(ivar*num_c)) = transpose(out(:,ivar));
    end
    
end


p_mean = zeros(var_num,num_c);

for ivar = 1 : var_num

     matrix_range = (1+(ivar-1)*num_c) : (ivar*num_c);
 
     cc = 0;
     for k = matrix_range %1 : num_c
         cc = cc + 1;          

         p_mean(ivar,cc) = mean(p(:,k));

         p(:,k) = p(:,k) - p_mean(ivar,cc);

     end

end

filename_m = [fname_raw, '_formatted.dat'];

OutputTecASCIIdata(filename_m,2,dimension_option,var_name,node_num,element_num,zone_type,out_format_xyz,out_format,out_format_c2n,xyz,p_mean',c2n);

end
end