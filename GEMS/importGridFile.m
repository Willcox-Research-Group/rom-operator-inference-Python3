function [xyz,c2n] = importGridFile(grid_file_name,line_skip_grid,read_format_xyz,read_format_c2n,ndim,element_num,node_num)
    
    fid = fopen(grid_file_name);
    
    for k = 1 : line_skip_grid
        fgetl(fid);
    end
    
    N = ceil((ndim+1) * node_num);
    
    import = fscanf(fid,read_format_xyz,N);
    
    xyz = zeros(node_num,ndim);
    
    for j = 1 : ndim
       
        range = ( 1 + node_num * ( j - 1 ) ) : node_num * j;
        
        xyz(:,j) = import(range);
        
    end
    
    c2n = zeros(element_num,2^ndim);
    
    import2 = fscanf(fid,read_format_c2n);
    
    for ni = 1 : element_num
        
        range = ( 1 + 2^ndim * ( ni - 1 ) ) : ni * 2^ndim;
        
        c2n(ni,:) = transpose(import2(range));
        
    end
    
    fclose(fid);

end
