clear all
clc
N=5:5:50;
L = 100;
num = 1000;
var_b_dB = -80;
var_b = 10^(var_b_dB/10)/1000;
var_w_dB = -80;
var_w = 10^(var_w_dB/10)/1000;
e3 = 0.1;
P_max_db = 36;
P_max = 10^(P_max_db/10)/1000;

% reference distance
d_ar = sqrt(100^2+25);
d_ab = sqrt(70^2+50);
d_aw = sqrt(100^2+50);
d_rb = sqrt(900+100+25);
d_rw = sqrt(125);

% pass loss exponent
a_ar = 2.4;
a_ab = 4.2;
a_aw = 4.2;
a_rb = 3;
a_rw = 3;

c0_dB = -30;
c0 = 10^(c0_dB/10);

% Rician factor
b_ar = 5;

%baseband equivalent channel from Alice to Bob
hd_ab = (randn(1,1) + 1i*randn(1,1))/sqrt(2);
h_ab = sqrt(c0*(1/d_ab)^a_ab)*hd_ab;

%baseband equivalent channel from Alice to Willie
hd_aw = (randn(1,1) + 1i*randn(1,1))/sqrt(2);
h_aw = sqrt(c0*(1/d_aw)^a_aw)*hd_aw;

%baseband equivalent channel from Alice to IRS
g_NLoS = (randn(50,1) + 1i*randn(50,1))/sqrt(2*50);
g_LoS = ones(50,1);
hd_ar = sqrt(b_ar/(b_ar+1))*g_LoS + sqrt(1/(b_ar+1))*g_NLoS;
har = sqrt(c0*(1/d_ar)^a_ar)*hd_ar;

%baseband equivalent channel from IRS to Bob
hd_rb = (randn(1,50) + 1i*randn(1,50))/sqrt(2*50);
hrb = sqrt(c0*(1/d_rb)^a_rb)*hd_rb;

%baseband equivalent channel from IRS to Willie
hd_rw = (randn(1,50) + 1i*randn(1,50))/sqrt(2*50);
hrw = sqrt(c0*(1/d_rw)^a_rw)*hd_rw;

sum_UB = zeros(1,length(N));
sum_PSCA = zeros(1,length(N));
sum_LC = zeros(1,length(N));
sum_noIRS = zeros(1,length(N));
for runs = 1:num
    for pr = 1:length(N)
        N1=N(pr);
        
        h_ar=har(1:N1,1);
        h_rb=hrb(1,1:N1);
        h_rw=hrw(1,1:N1);
        
        %%%%%%%%%% Upper bound %%%%%%%%%%
        b = diag(h_rb)*h_ar;
        b1 = b*b';
        b2 = b.*conj(h_ab);
        b3 = h_ab*b';
        b4 = (abs(h_ab)).^2;
        B = [b1 b2;b3 b4];
        
        a = diag(h_rw)*h_ar;
        a1 = a*a';
        a2 = a.*conj(h_aw);
        a3 = h_aw*a';
        a4 = (abs(h_aw)).^2;
        A = [a1 a2;a3 a4];
        
        cvx_begin sdp quiet
        variable W(N1+1,N1+1) hermitian %semidefinite
        variable Pa
        maximize real(trace(B*W))
        subject to
        Pa <= P_max
        L*trace(A*W) <= 2*e3^2*(trace(A*W) + var_w);
        W(N1+1,N1+1) == Pa;
        for n = 1:N1
            W(n,n) <= Pa;
        end
        W == hermitian_semidefinite(N1+1);
        cvx_end
        
        [U,Lam] = eig(W/Pa);
        theta_l = 2*pi*rand(N1+1,1);
        el = exp(1i*theta_l);
        ss = U*Lam^0.5*el;
        s = ss/ss(N1+1);
        q1 = exp(1i*angle(s(1:N1)));
        Q = diag(q1);
        
        SNR_UB(pr) = Pa/var_b*(abs(h_rb*Q*h_ar+h_ab))^2;
        SNR_UB_db(pr) = 10*log10(SNR_UB(pr));
        sum_UB(pr) = sum_UB(pr)+SNR_UB_db(pr);
        %%%%%%%%%% end Upper bound %%%%%%%%%%
        
        %%%%%%%%%% PSCA algorithm =1 %%%%%%%%%%
        error = 1;
        max_error = 0.0001;
        tau = 0.0001;
        c = 2;
        tau_max = 0.01;
        W0 = W;
        SNR0 = 0.000001;
        
        while(error>max_error)
            [U,Lam] = eig(W0);
            eigvalue = diag(Lam);
            lamda=max(eigvalue);
            for i=1:length(Lam)
                if lamda == eigvalue(i)
                    break
                end
            end
            w_max = U(:,i);
            
            cvx_begin sdp quiet
            variable W1(N1+1,N1+1) hermitian %semidefinite
            variable Pa
            variable eta nonnegative
            maximize trace(B*W1)-tau*eta
            subject to
            Pa<=P_max;
            L*trace(A*W) <= 2*e3^2*(trace(A*W) + var_w);
            W1(N1+1,N1+1) == Pa;
            for n = 1:N1
                W1(n,n) <= Pa;
            end
            W1 == hermitian_semidefinite(N1+1);
            trace(W1)-trace(w_max*w_max'*W1)<=eta;
            cvx_end
            
            [U,Lam] = eig(W1/Pa);
            theta_l = 2*pi*rand(N1+1,1);
            el = exp(1i*theta_l);
            ss = U*Lam^0.5*el;
            s = ss/ss(N1+1);
            q1 = exp(1i*angle(s(1:N1)));
            Q = diag(q1);
            
            tau1 = min(c*tau,tau_max);
            
            SNR1=Pa/var_b*(abs(h_rb*Q*h_ar+h_ab))^2;
            
            error=abs((SNR1-SNR0)/SNR0);
            SNR0=SNR1;
            
            W0=W1;
            tau=tau1;
        end
        
        SNR_PSCA_db(pr)=10*log10(SNR0);
        sum_PSCA(pr) = sum_PSCA(pr)+SNR_PSCA_db(pr);
        %%%%%%%%%% end PSCA algorithm =1 %%%%%%%%%%
        
        %%%%%%%%%% Low-complexity %%%%%%%%%%
        error = 1;
        max_error = 0.0001;
        u0 = ones(N1+1,1);
        P0 = P_max;
        
        [U6,Lam6] = eig(A);
        eigvalue6 = diag(Lam6);
        lamda6 = max(eigvalue6);
        I_N1_1 = eye(N1+1);
        
        a_1 = [a' conj(h_aw)]';
        
        while(error>max_error)
            f=(B/(u0'*A*u0)-(A-lamda6*I_N1_1)*(u0'*B*u0)/(u0'*A*u0)^2)*u0;
            s2 = u0'*B*u0/(u0'*A*u0)-2*a_1'*a_1*(N1+1)*(u0'*B*u0)/(u0'*A*u0)^2;
            
            cvx_begin sdp quiet
            variable u(N1+1,1) complex
            maximize real(2*f'*u+s2)
            subject to
            for x=1:N1+1
                abs(u(x,1))<=1;
            end
            %abs(u(N1+1,1))==1;
            cvx_end
            
            Pa_op=var_w*(e3^2+sqrt(e3^4+2*e3^2*L))/L*u'*A*u;
            error=(Pa_op-P0)/P0;
            P0=Pa_op;
            u0=u;
        end
        
        Pa_op=min(P0,P_max);
        
        [U,Lam] = eig(u*u');
        theta_l = 2*pi*rand(N1+1,1);
        el = exp(1i*theta_l);
        ss = U*Lam^0.5*el;
        s = ss/ss(N1+1);
        q1 = exp(1i*angle(s(1:N1)));
        Q = diag(q1);
        
        SNR_LC=Pa_op/var_b*(abs(h_rb*Q*h_ar+h_ab))^2;
        SNR_LC_db(pr)=10*log10(SNR_LC);
        sum_LC(pr) = sum_LC(pr)+SNR_LC_db(pr);
        %%%%%%%%%% end Low-complexity %%%%%%%%%%
        
        %%%%%%%%%% without IRS %%%%%%%%%%
        syms x
        Pa=vpasolve(exp(2*e3^2/L+x*(abs(h_aw))^2/(x*(abs(h_aw))^2+var_w))==1+x*(abs(h_aw))^2/var_w,x);
        Pa=min(Pa,P_max);
        
        SNR_noIRS(pr) = Pa/var_b*(abs(h_ab))^2;
        SNR_noIRS_db(pr) = 10*log10(SNR_noIRS(pr));
        sum_noIRS(pr) = sum_noIRS(pr)+SNR_noIRS_db(pr);
        %%%%%%%%%% end without IRS %%%%%%%%%%
    end
end

SNR_UB=real(sum_UB/num)
SNR_PSCA=real(sum_PSCA/num)
SNR_LC=real(sum_LC/num)
SNR_noIRS=real(sum_noIRS/num)
