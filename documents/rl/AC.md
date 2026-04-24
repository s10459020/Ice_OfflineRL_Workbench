# Actor Critic

## formula
nabla_J       = nabla_log_pi * A(s,a)
nabla_loss_pi = -nabla_log_pi * A(s,a)
Loss_pi       = -log_pi * A 

target_Q      => r + gamma * Q_theta(s',a')               # bellman
              => r + gamma * V_theta(s')                  # discrete
              => r + gamma * V_theta(s')                  # continuous
nabla_loss_Q  = (Q - target) * nabla_Q
Loss_Q        = 1/2( Q - target_Q )^2

target_V      => r + gamma * V_theta(s')                  # bellman
              =>       sum_a[   pi(a|s)*  Q_theta(s,a)  ] # discrete
              => 1/N * sum[ sample_pi[a]{ Q_theta(s,a)} ] # continuous
nabla_loss_V  = (V - target) * nabla_V
Loss_V        = 1/2( V - target_V )^2

nabla_log_pi, nabla_Q, nabla_V 視各實作而定

## Derivation

### Objective
Objective: __max_pi__ J(pi)
J(pi) = E_pi(tau){ R(tau) }
      = E_pi(tau){ sum[ gamma*r ] }

### Actor Update(Ascent)
Objective: max_pi __J(pi)__
Gradient Ascent: theta = theta + alpha * __nabla_theta J(pi)__

J(pi) =     E_pi(tau){ R(tau) }
      = sum[ pi(tau) * R(tau) ] # E{f} > sum[p*f]

nabla_J(pi) = d/d{theta}{ J(pi) }
            = d/d{theta}{ sum[ pi(tau) * R(tau) ] }
            =       sum[           d/d{theta}{         pi(tau)  } * R(tau) ] # R(tau)不依賴theta
            =       sum[ pi(tau) * d/d{theta}{     log_pi(tau)  } * R(tau) ] # log trick
            = E_pi[tau]{           d/d{theta}{     log_pi(tau)  } * R(tau) } # sum[p*f] > E{f}
            = E_pi[tau]{           d/d{theta}{ sum[log_pi(a|s)] } * R(tau) } 
                  #     pi(tau) =     p(s0) *   ∏{     pi(a|s) *     T(s'|s,a) }
                  # log pi(tau) = log p(s0) + sum{ log_pi(a|s) + log T(s'|s,a) }
                  # p(s0)與T(s'|s,a)微分為0
            = E_pi[tau]{sum_t[d/d{theta}{ log_pi(a|s) }] * R(tau)  } # REINFORCE
            = E_pi[tau]{sum_t[nabla_log_pi * G_t    ]} # R(tau) = ~~(r0 + r1 + ...)~~ + G_t
            = E_pi[tau]{sum_t[nabla_log_pi * Q(s,a) ]} # Q(s​,a​) = E_pi​{ G_t ​∣ s​,a​ }
            = E_pi[tau]{sum_t[nabla_log_pi * (Q-V)  ]} # E[ nabla_theta ​log(pi) V(s) ] = 0
            = E_pi[tau]{sum_t[nabla_log_pi * A(s,a) ]} 
           => nabla_log_pi * A(s,a) # 單步樣本

nabla log(pi) 需要視各模型實現決定


### Actor Update(Descent)
Objective: min{ __Loss_pi__ } = min{ __-J(pi)__ }
Gradient Descent: theta = theta - alpha * __nabla Loss_pi__

理論 Loss_pi = -J(pi)
理論 nabla_loss_pi = d/d{theta}{ Loss_pi }
                   = d/d{theta}{ -sum[ pi(tau) * R(tau) ] }
                   = - d/d{theta}{ sum[ pi(tau) * R(tau) ] }
                   = - nabla_J
                  => - nabla_log_pi * A(s,a) # 單步樣本

實作 Loss_pi = -E_pi[tau]{ sum[log_pi * A] }
            => -log_pi * A  # 單步樣本
實作 nabla_loss_pi = d/d{theta}{ -E_pi[tau]{ sum[log_pi * A] } }
                   = -E_pi[tau]{sum[ d/d{theta}{ log_pi } * A ]}
                   = - nabla_J   # 同理論
                  => - nabla_log_pi * A(s,a) # 單步樣本


### Advantage
nabla_theta J(pi) = E{sum_t[nabla_log_pi  * __A(s,a)__ ]}  
A(s,a) = __Q(s,a)__ - __V(s)__

V(s)   ~= __V_theta(s)__
        = E_pi[a],T[s']{ r + gamma * V(s') }       # bellman
        = E_pi[a]{ Q(s,a) }                        # build by Q
       =>       sum_a[   pi(a|s)*  Q_theta(s,a)  ] # discrete
      ~=> 1/N * sum[ sample_pi[a]{ Q_theta(s,a)} ] # continuous

Q(s,a) ~= __Q_theta(s,a)__
        = E_T[s'],pi[a']{ r + gamma * Q(s',a') }  # bellman
        = E_T[s']{ r + gamma * V(s') }            # build by V
       => r + gamma * V_theta(s')                 # discrete
       => r + gamma * V_theta(s')                 # continuous


### Critic(Q) update
Bellman: Q(s,a) = E{ r + gamma * Q(s',a') } = E{ target } 
Objective: min{ MSE[ Q(s,a) - target ] } = min{ __Loss_Q__ }
Gradient Descent: theta = theta - alpha * __nabla Loss_Q__

Loss_Q = MSE[ Q - target ]
       = 1/2( Q - target )^2

nabla Loss_Q = d/d{theta}{ Loss_Q }
             = d/d{theta}{ 1/2( Q - target )^2 }
             = (Q - target) * d/d{theta}{ Q - target } # chain rule
             = (Q - target) * d/d{theta}{ Q }          # semi-gradient TD 
             = (Q - target) * nabla_Q 

nabla_Q 需要視各模型實現決定


### Critic(V) update
Bellman: V(s) = E{ r + gamma * V(s') } = E{ target }
Objective: min{ MSE[ V(s) - target ] } = min{ __Loss_V__ }
Gradient Descent: theta = theta - alpha * __nabla Loss_V__

Loss_V = MSE[ V - target ]
       = 1/2( V - target )^2

nabla Loss_V = d/d{theta}{ Loss_V }
             = d/d{theta}{ 1/2( V - target )^2 }
             = (V - target) * d/d{theta}{ V - target } # chain rule
             = (V - target) * d/d{theta}{ V }          # semi-gradient TD
             = (V - target) * nabla_V 

nabla_V 需要視各模型實現決定