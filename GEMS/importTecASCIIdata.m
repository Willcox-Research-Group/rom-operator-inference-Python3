function out = importTecASCIIdata(file_name,line_skip,read_format,var,var_cc,var_node,node_num,element_num,c2n,ndim)


    fid = fopen(file_name);
    
    for k = 1 : line_skip
        fgetl(fid);
    end
    
    var_num_cc = length(var_cc);
    
    var_num_node = length(var_node);
    
    var_num = length(var);
    
    if ndim < 3
        N = ceil(var_num_cc * element_num + var_num_node * node_num);
        if isempty(var_cc) == 0
            out = zeros(element_num,var_num);
        else
            out = zeros(node_num,var_num);
        end
    else
        N = ceil(var_num_cc * element_num + var_num_node * node_num);
        out = zeros(element_num,var_num);
    end
    
    import = fscanf(fid,read_format,N);
    
%    rs = 1;
    
    for vp = 1 : var_num
        
        if ismember(vp,var_cc)
            increment = element_num;
        else
            increment = node_num;
        end
 
%        if vp == var(1)
         rs = 1 + increment*(var(vp)-1);
%        end
       
        re = rs + increment - 1;
        
        range = rs : re ;
        
        read_in = import(range);
        
        if ndim < 3
            out(:,vp) = read_in;
        else
            if ismember(var(vp),var_cc)
                out(:,vp) = read_in;
            else
                out(:,vp) = CalCellCenterValue(read_in,c2n,element_num);
            end
        end
        
%        rs = rs + increment;
    end
    
    fclose(fid);
end

function out = CalCellCenterValue(read_in,c2n,element_num)

    out = zeros(element_num,1);
    
    for ni = 1 : element_num
       
        nodes = c2n(ni,:);
        out(ni) = mean(read_in(nodes));
        
    end

end
