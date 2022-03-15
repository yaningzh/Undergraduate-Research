clear all
clc
N=50;
M=4;
num = 100;
d = 50;
var_k_dB = -80;
var_k = 10^(var_k_dB/10)/1000;
tar_SNR = 10;

% reference distance
d_AI = 1:7:50;
dv = 2;
d_AU = sqrt(d^2+dv^2);
d_IU = sqrt((d_AI-d).^2+dv^2);
D0=1;

% pass loss exponent
a_AI = 2;
a_IU = 2.8;
a_AU = 3.5;

c0_dB = -30;
c0 = 10^(c0_dB/10);

% Rician factor
b_AI = inf;
b_IU = 0;
b_AU = 0;

hd_NLoS = (randn(1,M) + 1i*randn(1,M))/sqrt(2*M); % Rayleigh fading component
hd_LoS = ones(1,M); %LOS
hd_AU = sqrt(b_AU/(b_AU+1))*hd_LoS + sqrt(1/(b_AU+1))*hd_NLoS; % Ricean Fading AU
h_AU = sqrt(c0*(D0/d_AU)^a_AU)*hd_AU;

g_NLoS = (randn(N,M) + 1i*randn(N,M))/sqrt(2*N); % Rayleigh fading component
g_LoS = ones(N,M); %LOS
g_AI = g_LoS; % Ricean Fading AI

hr_NLoS = (randn(1,N) + 1i*randn(1,N))/sqrt(2*N); % Rayleigh fading component
hr_LoS = ones(1,N); % LOS
g_IU =  sqrt(b_IU/(b_IU+1))*hr_LoS + sqrt(1/(b_IU+1))*hr_NLoS; % Ricean Fading IU

sum_LB = zeros(1,length(d_AI));
sum_SDR = zeros(1,length(d_AI));
sum_AO = zeros(1,length(d_AI));
sum_AU_MRT = zeros(1,length(d_AI));
sum_AI_MRT = zeros(1,length(d_AI));
sum_randPh = zeros(1,length(d_AI));
sum_noIRS = zeros(1,length(d_AI));

nn = rand(1,1)*N;
nn = ceil(nn);

for runs = 1:num
    for pr = 1:length(d_AI)
        
        h_AI = sqrt(c0*(D0/d_AI(pr))^a_AI)*g_AI;
        h_IU = sqrt(c0*(D0/d_IU(pr))^a_IU)*g_IU;
        
        %%%%%%%%%% without IRS %%%%%%%%%%
        w_noIRS = h_AU'/norm(h_AU);
        P_noIRS(pr) = tar_SNR*var_k/(norm(h_AU*w_noIRS))^2;
        P_noIRS_dBm(pr) = 30+10*log10(P_noIRS(pr));
        sum_noIRS(pr) = sum_noIRS(pr) + P_noIRS_dBm(pr);
        %%%%%%%%%% end without IRS %%%%%%%%%%
        
        
        %%%%%%%%%% semidefinite relaxation (SDR) %%%%%%%%%%
        phi = diag(h_IU)*h_AI;
        r1 = phi*(phi)';
        r2 = phi*(h_AU)';
        r3 = h_AU*(phi)';
        r4 = 0;
        R = [r1 r2;r3 r4];
        
        cvx_begin sdp quiet
        variable V(N+1,N+1) hermitian %semidefinite
        maximize real(trace(R*V) + (norm(h_AU))^2)
        subject to
        for n = 1:N
            V(n,n) == 1;
        end
        V == hermitian_semidefinite(N+1);
        cvx_end
        
        [U,Lam] = eig(V);
        theta_l = 2*pi*rand(N+1,1);
        el = exp(1i*theta_l);
        ss = U*Lam^0.5*el;
        s = ss/ss(N+1);
        q1 = exp(1i*angle(s(1:N)));
        Q = diag(q1);
        
        P_SDR(pr) = tar_SNR*var_k/(norm((h_IU*Q*h_AI+h_AU)))^2;
        P_SDR_dBm(pr) = 30+10*log10(P_SDR(pr));
        sum_SDR(pr) = sum_SDR(pr) + P_SDR_dBm(pr);
        %%%%%%%%%% end SDR %%%%%%%%%%
        
        
        %%%%%%%%%% alternating optimization %%%%%%%%%%
        w_AU = (h_AU)'/norm((h_AU)');
        phi_0 = angle(h_AU*w_AU);
        
        qq = h_IU;
        for o=1:N
            zeta1(o) = phi_0-angle(qq(1,o))-angle(h_AI(o,:)*w_AU);
        end
        
        Q1 = diag(exp(i*zeta1));
        w_AO = (h_IU*Q1*h_AI+h_AU)'/norm(h_IU*Q1*h_AI+h_AU);
        
        P_AO(pr) = tar_SNR*var_k/(norm((h_IU*Q1*h_AI+h_AU)*w_AO))^2;
        P_AO_dBm(pr) = 30+10*log10(P_AO(pr));
        sum_AO(pr) = sum_AO(pr) + P_AO_dBm(pr);
        %%%%%%%%%% end alternating optimization %%%%%%%%%%
        
        
        %%%%%%%%%% AP-user MRT %%%%%%%%%%
        P_AU_MRT(pr) = tar_SNR*var_k/(norm((h_IU*Q1*h_AI+h_AU)*w_AU))^2;
        P_AU_MRT_dBm(pr) = 30+10*log10(P_AU_MRT(pr));
        sum_AU_MRT(pr) = sum_AU_MRT(pr) + P_AU_MRT_dBm(pr);
        %%%%%%%%%% end AP-user MRT %%%%%%%%%%
        
        
        %%%%%%%%%% AP-IRS MRT %%%%%%%%%%
        g = h_AI(nn,:);
        w_AI = g'/norm(g');
        
        for o=1:N
            zeta2(o) = phi_0-angle(qq(1,o))-angle(h_AI(o,:)*w_AI);
        end
        
        Q2 = diag(exp(i*zeta2));
        P_AI_MRT(pr) = tar_SNR*var_k/(norm((h_IU*Q2*h_AI+h_AU)*w_AI))^2;
        P_AI_MRT_dBm(pr) = 30+10*log10(P_AI_MRT(pr));
        sum_AI_MRT(pr) = sum_AI_MRT(pr) + P_AI_MRT_dBm(pr);
        %%%%%%%%%% end AP-IRS MRT %%%%%%%%%%
        
        
        %%%%%%%%%% Random phase shift %%%%%%%%%%
        zeta3 = 2*pi*rand(1,N);
        Q3 = diag(exp(i*zeta3));
        w3 = (h_IU*Q3*h_AI+h_AU)'/norm(h_IU*Q3*h_AI+h_AU);
        
        P_randPh(pr) = tar_SNR*var_k/(norm((h_IU*Q3*h_AI+h_AU)*w3))^2;
        P_randPh_dBm(pr) = 30+10*log10(P_randPh(pr));
        sum_randPh(pr) = sum_randPh(pr) + P_randPh_dBm(pr);
        %%%%%%%%%% end Random phase shift %%%%%%%%%%
    end
end

P_LB = sum_AO/num;
P_SDR = sum_AO/num;
P_AO = sum_AO/num;
P_AU_MRT = sum_AU_MRT/num;
P_AI_MRT = sum_AI_MRT/num;
P_randPh = sum_randPh/num;
P_noIRS = sum_noIRS/num;

figure
grid on
plot(d_AI,P_LB,'-mo');
hold on
plot(d_AI,P_SDR,'-g');
hold on
plot(d_AI,P_AO,'--b');
hold on
plot(d_AI,P_AU_MRT,'-.r^');
hold on
plot(d_AI,P_AI_MRT,'-.cv');
hold on
plot(d_AI,P_randPh,':kp');
hold on
plot(d_AI,P_noIRS,':ks');
xlabel('AP-user horizontal distance, d(m)');
ylabel('Tramsmit power at the AP(dBm)');
legend('lower bound', 'SDR', 'Alternating optimization','AP-user MRT','AP-IRS MRT','Random phase shift','Without IRS');