function OutputTecASCIIdata(file_out,decomposition_option,ndim,var_name,node_num,element_num,zone_type,out_format_xyz,out_format,out_format_c2n,xyz,fsd_mean,c2n)

global var_per_line

    fid = fopen(file_out,'wt');
    fprintf(fid,'TITLE = "flame_surface_density.dat"\n');
    fprintf(fid,'VARIABLES = "x"\n');
    fprintf(fid,'"y"\n');
    if ndim > 2
        fprintf(fid,'"z"\n');
    end
    
    switch(decomposition_option)
        case(0)
            var_num = length(fsd_mean(1,:));
            for jj = 1 : var_num
                fprintf(fid,['"' strtrim(char(var_name(jj))) '"\n']);
            end
        case(1)
            fprintf(fid,['"real_' var_name '"\n']);
            fprintf(fid,['"imag_' var_name '"\n']);
        case(2)
            var_num = length(fsd_mean(1,:));
            for jj = 1 : var_num
                fprintf(fid,['"' strtrim(var_name(jj,:)) '"\n']);
            end
    end
    
    fprintf(fid,'ZONE T="zone 1"\n');
    fprintf(fid,'STRANDID=0, SOLUTIONTIME=0\n');
    fprintf(fid,['Nodes=' num2str(node_num) ', Elements=' num2str(element_num) ', ZONETYPE=' zone_type '\n']);
    fprintf(fid,'DATAPACKING=BLOCK\n');
    
    if ndim > 2 || length(fsd_mean(:,1)) < length(xyz(:,1))
        switch(decomposition_option)
            case(0)
                if var_num > 1
                   fprintf(fid,['VARLOCATION=([' num2str(ndim+1) '-' num2str(ndim+var_num) ']=CELLCENTERED)\n']);
                else
                   fprintf(fid,['VARLOCATION=([' num2str(ndim+1) ']=CELLCENTERED)\n']);
                end
            case(1)
                fprintf(fid,['VARLOCATION=([' num2str(ndim+1) '-' num2str(ndim+2) ']=CELLCENTERED)\n']);
            case(2)
                if var_num > 1
                   fprintf(fid,['VARLOCATION=([' num2str(ndim+1) '-' num2str(ndim+var_num) ']=CELLCENTERED)\n']);
                else
                   fprintf(fid,['VARLOCATION=([' num2str(ndim+1) ']=CELLCENTERED)\n']);
                end
        end
    end
    fprintf(fid,'DT=(SINGLE SINGLE SINGLE SINGLE SINGLE )\n');

    for j = 1 : ndim
        fprintf(fid,out_format_xyz,xyz(:,j)');
        if mod(length(xyz(:,j)),var_per_line) ~= 0
            fprintf(fid,'\n');
        end
    end
    
    if decomposition_option == 1
        fprintf(fid,out_format,transpose(real(fsd_mean)));
        fprintf(fid,out_format,transpose(imag(fsd_mean)));
    else
%        if decomposition_option == 2
%            fprintf(fid,out_format,transpose(fsd_mean));
%        else
            for jj = 1 : var_num
                fprintf(fid,out_format,transpose(fsd_mean(:,jj)));
                fprintf(fid,'\n');
            end
%        end
    end

    fprintf(fid,out_format_c2n,c2n');
    
    fclose(fid);

end
