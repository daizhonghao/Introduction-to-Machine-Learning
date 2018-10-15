y_label = h5read('train_labeled.h5','/train/block1_values')'+1;
label_arr = zeros(9000,10);
x_label = h5read('train_labeled.h5','/train/block0_values')';
x_unlabel = h5read('train_unlabeled.h5','/train/block0_values')';
x_test = h5read('test.h5','/test/block0_values')';
x_train = [x_label;x_unlabel];
x_train =  normalize(x_train);%,'center','mean');
x_test = normalize(x_test);
x_label  = x_train(1:9000,:);

for i =1:9000
    label_arr(i,y_label(i,1)) = 1;
end

% intialization
mu = [];
sigma = [];
for i =1:10
   mu(i,:) =  mean(x_label(y_label==i,:));
   n = size(x_label(y_label==i,:),1);
   ax = 0;
   x_sub = x_label(y_label==i,:);
   w(1,i) = n/9000;
   
   for j=1:n
        ax = (x_sub(j,:)-mu(i,:))'* (x_sub(j,:)-mu(i,:))+ax;
   end
   sigma(:,:,i) = ax/n;
   M = sigma(:,:,i);
   M_new = M + eye(size(M))*(-2*min(eig(M)));
   sigma(:,:,i) = M;
   
end

% EM%
rj = zeros(9000+21000,10);
maxiter = 1;

for kk=1:maxiter
    for i=1:10
    rj(:,i) = mvnpdf(x_train,mu(i,:),sigma(:,:,i));
    end
    rj = w.*rj;
    rj = rj./sum(rj,2);
    %rj(1:9000,1:10) = label_arr;
    w = sum(rj,1)/30000;
    for i=1:10
        mu(i,:) = sum(rj(:,i).*x_train,1)./sum(rj(:,i),1);
        ax = 0;
        for j=1:30000
        ax = (x_train(j,:)-mu(i,:))'* (x_train(j,:)-mu(i,:))*rj(j,i)+ax;
        end
        sigma(:,:,i) = ax/sum(rj(:,i),1);
    end

kk
end

%% predict
% y_pre = zeros(30000,10);
% for i =1:10
%     y_pre(:,i) = w(1,i)*mvnpdf(x_train_new,mu(i,:),sigma(:,:,i));
% end


