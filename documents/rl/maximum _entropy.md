# Entropy Regularization 

## formula
Loss_pi  = alpha* log_pi - Q 

target_Q => r + gamma* ( Q2 - alpha* log_pi2 )    # bellman
         => r + gamma * V2_theta                  # discrete
         => r + gamma * V2_theta                  # continuous
Loss_Q   = 1/2( Q - target_Q )^2

## Derivation
在SAC被提出，增加loss_entropy，使行為高熵(避免行為太單一)

### Objective
trick: r => r + __alpha * H[ pi(.|s) ]__  # update every step
H(pi) = -E_pi[a]{ log_pi(a|s) }           # entropy
      = -sum_a[ pi(a|s) * log_pi(a|s) ] 
Objective: max_pi E{sum[ gamma * r_soft ] }
         = max_pi E{sum[ gamma * (r + alpha * H(pi)) ]}

### Critic
在SAC版本中，不討論r => r_soft，而是使用V/Q的等價版本
Q_soft = E_T[s']{ r + gamma* V(s') }
V_soft = E_pi[a]{ Q_soft - alpha* log_pi(a|s) }

Q_soft ~= __Q_theta(s,a)__                                    # [參數化]
        = E_T[s']{ r + gamma * V_theta(s')}                   # SAC definition
        = r + gamma  * E_T[s']{ V_theta(s')}                  # 移出期望值
       => r + gamma * V_theta(s')                             # [單樣本discrete]
       => r + gamma * V_theta(s')                             # [單樣本continuous]
       => r + gamma* ( Q_soft(a'|s') - alpha* log_pi(a'|s') ) # [單樣本target]

V_soft ~= __V_theta(s)__                                      # [參數化]
        =     E_pi[a]{ Q_theta(s,a) - alpha* log_pi(a|s)  }   # SAC definition
       => sum_a[ pi * (Q_theta(s,a) - alpha* log_pi(a|s)) ]   # [單樣本discrete]
       ~=   1/N sum_N[ Q_theta(s,a) - alpha* log_pi(a|s)  ]   # [單樣本continuous]
       =>          r + gamma* V(s') - alpha* log_pi(a|s)      # [單樣本target]

       
### Critic update
Gradient Descent: theta = theta - alpha * __nabla Loss_V/Q_soft__

Loss_Q_soft = MSE[ Q_soft - target_Q_soft ]
            = 1/2( Q_soft - [r + gamma* ( Q_soft2 - alpha* log_pi2 )] )^2
            = 1/2( Q_soft -  r - gamma* ( Q_soft2 - alpha* log_pi2 ) )^2
            
Loss_V_soft = MSE[ V_soft - target_V_soft ]
            = 1/2( V_soft - [r + gamma* V2 - alpha* log_pi] )^2
            = 1/2( V_soft -  r - gamma* V2 + alpha* log_pi  )^2

### Actor update
Loss_pi_soft  = -E{ log_pi * A_soft } # AC loss