function [ Dic, train, test ] = split_data_cov_sirali(data_train, label_train, data_test, label_test, param)

class_n = length(unique(label_train)); %n umber of class
N1 = param.dictionary_size; 
N2 = (size(label_train, 1)/class_n) - N1;

%MS=param.matrixsize; %[19,2]
gds=[1; unique(cumprod(perms(factor(class_n)),2))];
m1=gds(end-1);
m2=class_n/m1;
gds=[1; unique(cumprod(perms(factor(N1)),2))];
m22=gds(floor((length(gds)-2)/2)+1);
m11=N1/m22;

m22 = 25;
m11 = 25;

%% initilization
D = zeros(size(data_train, 1), N1 * class_n);
train.data = zeros(size(data_train,1), N2 * class_n);
test.data =  zeros(size(data_test,1), size(data_test,2));
train.label = zeros(N2 * class_n, 1);
test.label = zeros(size(label_test, 1), 1);


temp= [];
A=[];
t=1;
for k=1:m2
    for l=1:m1
        temp=[temp,ones(m11,m22)*t];
        t=t+1;
    end
    A=[A;temp];
    temp=[];    
end

for i=1:class_n
    in=find(A(:)==i);
    in2=find(label_train==i);

    for k=1:length(in)
        D(:,in(k))=data_train(:,in2(k));
%        figure(99),imshow(reshape(D(:,k-1),32,32),[])
    end
    for l=1:N2
        train.data(:,N2*(i-1)+l) = data_train(:,in2(k+l));
        train.label(N2*(i-1)+l)=i;
    end
Dic.dictionary=D;
Dic.label=A(:);
Dic.label_matrix=A;
end

test.data = data_test;
test.label = label_test;
