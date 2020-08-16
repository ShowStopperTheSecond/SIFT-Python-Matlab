
close all;clc;clear;
% parameters

n_oct=8;
n_spo=3;
sigma_in=.5;
sigma_min=.8;
delta_in=1;
l_desc=6;
l_ori=1.5;
c_dog=.015;
r_edge=10;

img1=imread('./SIFT_Python/reeses_puffs.png');
img2=imread('./SIFT_Python/many_cereals.jpg');
img1=rgb2gray(img1);
img2=rgb2gray(img2);
% img1=double(img1);
% img2=double(img2);
img1=im2single(img1);
img2=im2single(img2);


kp1=siftDetectAndCompute(img1,n_oct,n_spo,sigma_in,sigma_min,delta_in,c_dog,r_edge,l_ori,l_desc)
kp2=siftDetectAndCompute(img2,n_oct,n_spo,sigma_in,sigma_min,delta_in,c_dog,r_edge,l_ori,l_desc)
save('kp1','kp1')
save('kp2','kp2')


load('kp1.mat');
load('kp2.mat');

matches=findMatches(kp1,kp2,.6);
draw_matches(matches,img1,img2)



drawKeyPoints(img1,kp1,"key points img1")
drawKeyPoints(img2,kp2,"key points img2")

function results=findMatches(kp1,kp2,ratio)
    
    
    desc1=zeros(length(kp1),4*4*8);
    for i=1:length(kp1)
        desc1(i,:)=kp1(i).descriptor;
    end
    
        
    desc2=zeros(length(kp2),4*4*8);
    for i=1:length(kp2)
        desc2(i,:)=kp2(i).descriptor;
    end
    
    match_kp1=[]
    match_kp2=[]
    [index,distance]=matchFeatures(desc1,desc2,'MatchThreshold',10);
    for i =1:length(index)
            match_kp1=[match_kp1 kp1(index(i,1))] 
            match_kp2=[match_kp2  kp2(index(i,2))];
    end
    
    results={match_kp1,match_kp2}
            
end
function draw_matches(matches,img1,img2)
    [img1_height,img1_width]=size(img1)
    [img2_height,img2_width]=size(img2)
    tmp_height=max(img1_height,img2_height)
    tmp_width=img1_width+img2_width
    tmp_img=zeros(tmp_height,tmp_width)
    tmp_img(1:img1_height,1:img1_width)=img1
    tmp_img(1:img2_height,img1_width+1:end)=img2
    imshow(tmp_img)
    for i=1:length(matches{1})
        pt1=[ round(matches{1}(i).y) round(matches{1}(i).x)]
        pt2=[ round(matches{2}(i).y)+img1_width round(matches{2}(i).x)]
        hold on
        drawline('Position',[pt1;pt2])
        
    end
    pause(1)
end


function result= siftDetectAndCompute(img_in,n_octave,n_spo,sigma_in,sigma_min,delta_in,c_dog,r_edge,l_ori,l_desc)
    fprintf("Starting SIFT\n")
    fprintf("Creating Gaussian Parymid\n")
    ss=createScaleSpace(double(img_in),n_octave,n_spo ,sigma_min,delta_in ,sigma_in);

%     displaySpace(ss)
    dog=computeDoG(ss);

    fprintf("Phase1: Detecting Extremums\n")
    candidate_points=detectExtremums(dog,sigma_min,delta_in,c_dog);
    length(candidate_points)
    title_text="Phase1: Extremums";
    drawKeyPoints(img_in,candidate_points,title_text)
    
    fprintf("Phase2: Discarding low contrast points (conservative filtering)\n")
    candidate_points=discardLowContrastPoints_conservative(candidate_points,c_dog);
    length(candidate_points);
    title_text="Phase2: PhaseDiscarding low contrast points (conservative filtering)";
    drawKeyPoints(img_in,candidate_points,title_text)
    
    fprintf("Phase3: Refining Position\n")
    candidate_points=refinePositions(candidate_points,dog,15,delta_in,sigma_min);
    length(candidate_points)
    title_text="Phase3: Refining Position";
    drawKeyPoints(img_in,candidate_points,title_text)
 
    fprintf("phase7: Descriptors Computed\n")
    discardLowContrastPoints(candidate_points,c_dog);
    length(candidate_points)
    title_text="Phase4: Discarding low contrast points (final filtering)"
    drawKeyPoints(img_in,candidate_points,title_text)
   
    fprintf("Phase5: Checking Edge Response\n")
    candidate_points=discardEdgeResponse(candidate_points,dog,r_edge,n_octave,n_spo,sigma_min,1);
    length(candidate_points)
    title_text="Phase5: Checking Edge Response"
    drawKeyPoints(img_in,candidate_points,title_text)
    
    fprintf("phase6: orientation_detection\n")
    g_ss=spaceGradient(ss)
    candidate_points=detectOrientation(candidate_points,g_ss,l_ori,36,.8,sigma_min,n_octave,n_spo,delta_in);
    length(candidate_points)
    title_text="phase6: orientation_detection";
    drawKeyPoints(img_in,candidate_points,title_text)
    
    fprintf("phase7: Descriptors Computed\n")
    candidate_points=computeDescrptor(candidate_points,g_ss,delta_in,l_desc,8,4,n_octave,n_spo,sigma_min);
    length(candidate_points)
    title_text="phase7: Descriptors Computed";
    drawKeyPoints(img_in,candidate_points,title_text)
    result=candidate_points;
    fprintf("The End")
end


function smooth_img = gaussian_smooth(img, sigma)

kernel_size = bitor(round(sigma*6 + 1), 1);
kernel = fspecial('gaussian', [kernel_size kernel_size], sigma);
smooth_img = imfilter(img,kernel,'replicate');

end


function drawKeyPoints(img,points,title_text)
    figure

    [img_height,img_width]=size(img);
    for i=1:length(points)
        m=round(points(i).x);
        n=round(points(i).y);
        r=round(points(i).sigma);
        for d=linspace(0,2*pi,round(r*50))
            a=floor(r*cos(d));
            b=floor(r*sin(d));
            if 0<m+b && m+b<img_height && n+a>0 && n+a<img_width
                img(m+b,n+a)=1;
            end
        end
%         drawcircle('Center',[n,m],'Radius',r,'InteractionsAllowed','none');

        ori=points(i).orientation;
        for d=linspace(0,r,round(10*r))
            a=floor(d*cos(ori));
            b=floor(d*sin(ori));
            if 0<m+b && m+b<img_height && n+a>0 && n+a<img_width
                img(m+b,n+a)=1;
            end
        end
        
    end
    imshow(img);
    title(title_text)
    pause(1)
end


function result=computeDescrptor(points,gradient_of_scale_space,delta_in,l_desc,n_ori,n_hist,n_octave,n_spo,sigma_min)
   
    total_blurring_map=generateBlurringMat(n_octave,n_spo,sigma_min,delta_in) ;
    total_blurring_map=total_blurring_map(:,2:end-2) ;
    total_blurring_map=total_blurring_map' ;
    total_blurring_map=total_blurring_map(:); 
    gradient_m=gradient_of_scale_space(:,1:n_spo+3) ;
    gradient_n=gradient_of_scale_space(:,n_spo+4:end) ;
    
    gradient_m=gradient_m(:,2:end-2) ;
    gradient_n=gradient_n(:,2:end-2) ;
    [h,w]=size(gradient_m{2,1}) ;
    result=[] ;
    counter=0;
    for index=1:length(points)
        
        sigma_dist=abs(total_blurring_map(:)-points(index).sigma) ;
        [~,closest_scale]=min(sigma_dist) ;
        octave=ceil(closest_scale/n_spo) ;
        scale=mod(closest_scale,n_spo)+n_spo*(mod(closest_scale,n_spo)==0) ;
        safety_dist=sqrt(2)*l_desc*points(index).sigma*(n_hist+1)/n_hist ;
        delta_octave=delta_in*2^(octave-2) ;
        
        if points(index).x<safety_dist || points(index).x>(h-safety_dist) ...
                ||points(index).y<safety_dist || points(index).y>(w-safety_dist)
            continue
        end
%         feature_desc=zeros(1,n_hist*n_hist*n_ori)
        patch_hist=zeros(n_hist,n_hist,n_ori) ;
%         r_outer_patch=safety_dist
        p_corn_m0= round((points(index).x-safety_dist)/delta_octave) ;
        p_corn_m1= round((points(index).x+safety_dist)/delta_octave) ;
        p_corn_n0= round((points(index).y-safety_dist)/delta_octave) ;
        p_corn_n1= round((points(index).y+safety_dist)/delta_octave) ;
        
%         inner_patch_width=2*l_desc*points(index).sigma*(n_hist+1)/n_hist 
        points_in_patch=0 
        for m =p_corn_m0:p_corn_m1
            for n=p_corn_n0:p_corn_n1
                x_dist=m*delta_octave-points(index).x;
                y_dist=n*delta_octave-points(index).y ;
                x_patch=(x_dist*cos(points(index).orientation)+y_dist*sin(points(index).orientation))/points(index).sigma ;
                y_patch=(-x_dist*sin(points(index).orientation)+y_dist*cos(points(index).orientation))/points(index).sigma ;
                
                if max(abs(x_patch),abs(y_patch)) <l_desc*(n_hist+1)/n_hist
                    points_in_patch=points_in_patch+1 ;
                    grad_m=gradient_m{octave,scale}(m,n) ;
                    grad_n=gradient_n{octave,scale}(m,n) ;
                    
                    gradient_magnitude=sqrt(grad_m^2+grad_n^2) ;
                    ori_cont=exp(-(x_patch^2+y_patch^2)/(2*(l_desc)^2))*gradient_magnitude;
                    ori=mod(atan2(grad_m,grad_n),2*pi) ;
                    ori=ori-points(index).orientation ;
                    ori=mod(ori,2*pi) ;
                    failed=true ;
                    for i=1:n_hist
                        for j=1:n_hist
                            x_i=(i-(1+n_hist)/2)*2*l_desc/n_hist ;
                            y_j=(j-(1+n_hist)/2)*2*l_desc/n_hist ;
                            if max(abs(x_i-x_patch),abs(y_j-y_patch))<=2*l_desc/n_hist
                                for k=1:n_ori
                                    ori_k=2*pi*k/n_ori ;
                                    angle_dis=abs(ori_k-ori) ;
                                    if angle_dis<2*pi/n_ori
                                        patch_hist(i,j,k)=patch_hist(i,j,k)+(1-n_hist/(2*l_desc)*abs(x_patch-x_i))* ... 
                                                                            (1-n_hist/(2*l_desc))*abs(y_patch-y_j)* ...
                                                                            (1-n_ori/(2*pi)*angle_dis)*ori_cont ;
                                        failed=false ;
                                    end
                                end
                            end
                        end
                    end
                    if failed; counter=counter+1 ;counter;end
                    
                end
                points_in_patch ;
            end
        end
        
        feature_desc=patch_hist(:);
        l2_feature_desc=norm(feature_desc) ;
        feature_desc=min(feature_desc,.5*l2_feature_desc);
        feature_desc=min(512*feature_desc/l2_feature_desc,255);
        
        points(index).descriptor=round(feature_desc) ;
        result=[result points(index)] ;
        length(result)
        fprintf("failed:" + counter)
        
    end
end


   
function result=detectOrientation(points,gradient_of_scale_space,l_ori,n_bins,t_senondary,sigma_min,n_octave,n_spo,delta_in)
    
    total_blurring_map=generateBlurringMat(n_octave,n_spo,sigma_min,delta_in);
    total_blurring_map=total_blurring_map(:,2:end-2);
    total_blurring_map=total_blurring_map';
    total_blurring_map=total_blurring_map(:);
    gradient_m=gradient_of_scale_space(:,1:n_spo+3);
    gradient_n=gradient_of_scale_space(:,n_spo+4:end);
    
    gradient_m=gradient_m(:,2:end-2);
    gradient_n=gradient_n(:,2:end-2);
    
    [plane_height,plane_width]=size(gradient_m{2,1});
    result=[];
    for i=1:length(points)
        hist=zeros(1,n_bins);
        sigma_dist=abs(total_blurring_map-points(i).sigma);
        [~,closest_scale]=min(sigma_dist(:));
        octave=ceil(closest_scale/n_spo);
        scale=mod(closest_scale,n_spo)+n_spo*(mod(closest_scale,n_spo)==0);
        safety_dist=3*l_ori*points(i).sigma;
        delta_octave=delta_in*2^(octave-2);
        
        if points(i).x<=safety_dist || points(i).x>=(plane_height-safety_dist) ...
                ||points(i).y<=safety_dist || points(i).y>(plane_width-safety_dist)
            continue
        end
        p_corn_m0= ceil((points(i).x-safety_dist)/delta_octave);
        p_corn_m1= floor((points(i).x+safety_dist)/delta_octave);
        p_corn_n0= ceil((points(i).y-safety_dist)/delta_octave);
        p_corn_n1= floor((points(i).y+safety_dist)/delta_octave);
        
        for m=p_corn_m0:p_corn_m1
            for n=p_corn_n0:p_corn_n1
                grad_m=gradient_m{octave,scale}(m,n);
                grad_n=gradient_n{octave,scale}(m,n);
                
                gradient_magnitude=sqrt(grad_m^2+grad_n^2);
                relative_m=m*delta_octave-points(i).x;
                relative_n=n*delta_octave-points(i).y;
                ori_cont=exp(-(relative_m^2+relative_n^2)/(2*(l_ori*points(i).sigma)^2))*gradient_magnitude;
                ori=mod(atan2(grad_m,grad_n),2*pi);
                ori_bin=mod(floor(n_bins/(2*pi)*ori),n_bins)+1;
                hist(ori_bin)=hist(ori_bin)+ori_cont;
            end
        end
        smoothed_hist=smoothHist(hist);
        hist_max=max(smoothed_hist(:));
        refrence_angle=[];
        for k=1:n_bins
            h_p=smoothed_hist( n_bins*(k==1)+(k-1)*(k~=1));
            h_c=smoothed_hist(k);
            h_n=smoothed_hist((k==n_bins)+(k~=n_bins)*(k+1));
            
            if h_c >h_n && h_c>h_p && h_c>t_senondary*hist_max
                angle_at_bin_k=2*pi*k/n_bins;
                key_ori=angle_at_bin_k+pi/n_bins*(h_p-h_n)/(h_p-2*h_c+h_n);
                refrence_angle=[refrence_angle key_ori];
            end
        end
        
        for index=1:length(refrence_angle)
            final_key=points(i);
            final_key.orientation=refrence_angle(index);
            result=[result final_key];
            length(result)
        end
        
    end
end

function result=smoothHist(hist)
    h_size=length(hist);
    result=zeros(1,length(hist));
    
    for repeat =1:6
        for i=1:h_size
            h_p=hist( h_size*(i==1)+(i-1)*(i~=1));
            h_c=hist(i);
            h_n=hist((i==h_size)+(i~=h_size)*(i+1));
            result(i)=(h_p+h_c+h_n)/3;
        end
    end
end


function result= spaceGradient(scale_space)
    [n_octave,n_img_per_octave]=size(scale_space);
    gradient_m=cell(n_octave,n_img_per_octave);
    gradient_n=cell(n_octave,n_img_per_octave);
    for octave=1:n_octave
        [img_height,img_width]=size(scale_space{octave,1});
        for scale=1:n_img_per_octave
            [gradient_m{octave,scale},gradient_n{octave,scale} ]=imgradientxy(scale_space{octave,scale});
        end
    end
     
    result=[gradient_m,gradient_n];
end

function result=discardEdgeResponse(points,DoG,r_edge,n_octave,n_spo,sigma_min,delta_in)
    result=[];
    total_blurring_map=generateBlurringMat(n_octave,n_spo,sigma_min,delta_in);
% %     Ask??
%     total_blurring_map=total_blurring_map(:,2:end-2);
%     DoG=DoG(:,2:end-1);
    for i=1:length(points)
        m=points(i).m;
        n=points(i).n;
        octave=points(i).octave;
        sigma_dist=abs(total_blurring_map(octave,:)-points(i).sigma);
        [~,closest_scale]=min(sigma_dist(:));
       
        scale=mod(closest_scale,n_spo)+n_spo*(mod(closest_scale,n_spo)==0);
        
        surface=DoG{octave,scale}(m-1:m+1,n-1:n+1);
        hessian=hessian2d(surface);
        hessian_trace=trace(hessian);
        hessian_det=det(hessian);
        edgness=hessian_trace^2/hessian_det;
        if edgness<(r_edge+1)^2/r_edge
            result=[ result points(i)];
        end
    end
end

function result=refinePositions(points,difference_of_gaussian_space,n_attempts,delta_in,sigma_min)
    [n_octave,n_img_per_octave]=size(difference_of_gaussian_space);
    n_spo=n_img_per_octave-2;
    result=[];
    for i=1:length(points)
        o=points(i).octave;
        s=points(i).scale;
        m=points(i).m;
        n=points(i).n;
        outside_bound=false;
        ignore=false;
        [img_height,img_Width]=size(difference_of_gaussian_space{o,1});
        for attempt=1:n_attempts
            if (m<2 || m > (img_height-1)) || (n<2 || n > (img_Width-1)) || (s <2 || s>n_img_per_octave-1)
                outside_bound=true;
                break
            end
            cube=cat(3,difference_of_gaussian_space{o,s-1}(m-1:m+1,n-1:n+1),difference_of_gaussian_space{o,s}(m-1:m+1,n-1:n+1),difference_of_gaussian_space{o,s+1}(m-1:m+1,n-1:n+1));
           
            
            cube=permute(cube,[3,1,2]);
            hessian=hessian3d(cube);
            gradient=gradient3d(cube);
            offset=-inv(hessian)*gradient';
            if all(isnan(offset))
                ignore=true;
                break
            end
            w=difference_of_gaussian_space{o,s}(m,n) +.5*gradient*offset;
            sigma=2^(o-1)*sigma_min*2^((offset(1)+s-1)/n_spo);
            x=delta_in*2^(o-2)*(offset(2)+m);
            y=delta_in*2^(o-2)*(offset(3)+n);
            s=s+round(offset(1));
            m=m+round(offset(2));
            n=n+round(offset(3));
            if all(abs(offset)<.5) ;break;end
            
        end
        if outside_bound || ignore ;continue;end
        if any(abs(offset)>1);continue;end
        points(i).x=x;
        points(i).y=y;
        points(i).w=w;
        points(i).sigma=sigma;
        result= [ result points(i)];
    end    

end


function result=generateBlurringMat(n_octave,n_spo,sigma_min,delta_in)
    k = 2 ^ (1 / n_spo);
    iteration_per_octave = 3 + n_spo;
    blurring_matrix = zeros(n_octave, iteration_per_octave);
    for octave = 1:n_octave
        iteration_sigma_min = 2 ^ (octave-1) * sigma_min / delta_in;
        blurring_matrix(octave, 1) = iteration_sigma_min;
        for iteration= 2: iteration_per_octave
            total_blur = k ^ (iteration-1) * iteration_sigma_min;
            blurring_matrix(octave, iteration) = total_blur;
        end
    end
    result=blurring_matrix;
    
end




function result=hessian2d(surface)
    h11=surface(3,2) +surface(1,2)-2*surface(2,2);
    h22=surface(2,3)+surface(2,1)-2*surface(2,2);
    h12=(surface(3,3)-surface(3,1)-surface(1,3)+surface(1,1))/4;
    
    result=[h11 h12
            h12 h22];
end

function result = gradient3d(cube)
    ds = 0.5 * (cube(3,2,2) - cube(1, 2,2));
    dm = 0.5 * (cube(2,3,2) - cube(2,1,2));
    dn = 0.5 * (cube(2,2,3) - cube(2,2,1));
    result= [ds dm dn];
end


function result=hessian3d(cube)

    center_point = cube(2, 2, 2);
    h11 = cube(3,2,2) - 2 * center_point + cube(1,2,2);
    h22 = cube(2,3,2) - 2 * center_point + cube(2,1,2);
    h33 = cube(2,2,3) - 2 * center_point + cube(2,2,1);
    h12 = 0.25 * (cube(3,3,2) - cube(3,1,2) -cube(1,3,2) +cube(1,1,2));
    h13 = 0.25 * (cube(3,2,3) - cube(3,2,1) -cube(1,2,3) +cube(1,2,1));
    h23 = 0.25 * (cube(2,3,3) - cube(2,3,1) -cube(2,1,3) +cube(2,1,1));
    
    result =[h11, h12, h13
            h12, h22, h23
            h13, h23, h33];

end




function result=discardLowContrastPoints(key_point_list,c_dog)
    result=[];
    n_points=length(key_point_list);
    for i=1:n_points
        if abs(key_point_list(i).w) > c_dog
            result=[result key_point_list(i)];
        end
    end
    
end



function result=discardLowContrastPoints_conservative(key_point_list,c_dog)
    result=[];
    n_points=length(key_point_list);
    for i=1:n_points
        if abs(key_point_list(i).w) > .8*c_dog
            result=[result key_point_list(i)];
        end
    end
    
end



function result=detectExtremums(difference_of_gaussian_space,sigma_min,delta_in,c_dog)
    key_point=struct('m',1,'n',1,'x',1,'y',1,'octave',1,'scale',1,'sigma',0,'w',0,'orientation',0,'descriptor',0);
    key_points_list=[];
    [n_octave,n_img_per_octave]=size(difference_of_gaussian_space);
    n_spo=n_img_per_octave-2;
    total_blurring_map=generateBlurringMat(n_octave,n_spo,sigma_min,delta_in);
   for octave=1:n_octave
       for img_index=2:n_img_per_octave-1
           
           [img_height,img_width]=size(difference_of_gaussian_space{octave,1});
           for m=2:img_height-1
               for n=2:img_width-1
                    cube_data=[ difference_of_gaussian_space{octave,img_index-1}(m-1:m+1,n-1:n+1)
                                difference_of_gaussian_space{octave,img_index}(m-1:m+1,n-1:n+1)
                                difference_of_gaussian_space{octave,img_index+1}(m-1:m+1,n-1:n+1)];
                    center_pt=difference_of_gaussian_space{octave,img_index}(m,n);
                    
                    if (center_pt==max(cube_data(:)) || center_pt==min(cube_data(:))) && center_pt>.8*c_dog
                        found_point=key_point;
                        found_point.m=m;
                        found_point.n=n;
                        found_point.x=m*delta_in*2^(octave-2);
                        found_point.y=n*delta_in*2^(octave-2);
                        found_point.sigma=total_blurring_map(octave,img_index+1);
                        found_point.octave=octave;
                        found_point.scale=img_index;
                        found_point.w=center_pt;
                        key_points_list= [key_points_list found_point];
                    end
               end
           end
       end
   end
   result=key_points_list;
                                
end


function result=computeDoG(scale_space)

    [n_octave,n_img_per_octave]=size(scale_space);
    result=cell(n_octave,n_img_per_octave-1);
    for octave=1:n_octave
        for scale=2:n_img_per_octave
            result{octave,scale-1}=scale_space{octave,scale}-scale_space{octave,scale-1};
        end
    end
end



function result=createScaleSpace(img_in,n_oct,n_spo,sigma_min,delta_in,sigma_in)

    delta_min=delta_in/2;
    scale_factor=1/delta_min;
    seed_img=imresize(img_in,scale_factor,'bilinear');
    blur2apply=sqrt(sigma_min^2-sigma_in^2)/delta_min;
    n_img_per_octave=n_spo+3;
    gaussian_scale_space=cell(n_oct,n_img_per_octave);
    gaussian_scale_space{1,1}=gaussian_smooth(seed_img,blur2apply);
    for s=2:n_img_per_octave
        blur2apply=sigma_min/delta_min*sqrt(2^(2*(s-1)/n_spo)-2^(2*(s-2)/n_spo));
        gaussian_scale_space{1,s}=gaussian_smooth(gaussian_scale_space{1,s-1},blur2apply);
    end
    
    for octave=2:n_oct
        gaussian_scale_space{octave,1}=imresize(gaussian_scale_space{octave-1,n_spo+1},.5,'bilinear');
        for s=2:n_img_per_octave
            blur2apply=sigma_min/delta_min*sqrt(2^(2*(s-1)/n_spo)-2^(2*(s-2)/n_spo));
            gaussian_scale_space{octave,s}=gaussian_smooth(gaussian_scale_space{octave,s-1},blur2apply);
        end
    end
    
    result=gaussian_scale_space;

end

function displaySpace(space)
    [n_oct,n_img_per_oct]=size(space);
    index=1;
    figure
    for octave=1:n_oct
        for scale=1:n_img_per_oct
           subplot( n_oct,n_img_per_oct,index);
%            imshow(normalize(space{octave,scale},'range'));
           imshow(space{octave,scale})
           title(size(space{octave,scale}));
           index=index+1;
        end
    end
    
end

