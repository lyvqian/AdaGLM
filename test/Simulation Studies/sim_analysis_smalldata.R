load("./test/res_binomial_small.Rda")
load("./test/res_gaussian_small.Rda")
load("./test/res_poisson_small.Rda")
load("./test/res_gamma_small.Rda")

library(ggplot2)
library(dplyr)

wide_df_poisson<-wide_df_poisson[wide_df_poisson$adadelta_MSE<10,]
wide_df_gamma<-wide_df_gamma[wide_df_gamma$adadelta_MSE<10,]

df <- as.data.frame(rbind(wide_df_binomial, wide_df_gaussian, wide_df_poisson, wide_df_gamma))
df$family <- c(rep("binomial", nrow(wide_df_binomial)),
               rep("gaussian", nrow(wide_df_gaussian)),
               rep("poisson", nrow(wide_df_poisson)),
               rep("gamma", nrow(wide_df_gamma)))

df_long <- df %>%
  pivot_longer(
    cols = ends_with("_MSE") | ends_with("_Time"),
    names_to = c("Method", "metrics"),
    names_sep = "_",
    values_to = "y_value"
  )

df_long$Method = factor(df_long$Method, levels=c("adagrad", "adadelta", "adam", "adasmooth","glm"))

ggplot(df_long[df_long$metrics=="MSE", ], aes(x = factor(family), y = y_value, fill=Method)) +  
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.7) +
  facet_wrap(~family, scales = "free", nrow=1) +
  guides(fill = guide_legend(byrow = TRUE, nrow = 1)) +
  labs(y = "MSE", fill = "Method", 
       x = "") +
  theme(legend.position = "bottom", 
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        plot.title = element_text(hjust=0.5, size=12),
        axis.title = element_text(size = 14),
        strip.text = element_text(size = 14, face = "bold"),
        legend.text = element_text(size = 12)) 

#ggsave("./test/simulate_smalldata.jpg", dpi=600)

ggplot(df_long[df_long$metrics=="Time", ], aes(x = factor(family), y = y_value, fill=Method)) +  
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.7) +
  facet_wrap(~family, scales = "free", nrow=1) +
  guides(fill = guide_legend(byrow = TRUE, nrow = 1)) +
  labs(y = "Time (ms)", fill = "Method", 
       x = "") +
  theme(legend.position = "bottom", 
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        plot.title = element_text(hjust=0.5, size=12),
        axis.title = element_text(size = 14),
        strip.text = element_text(size = 14, face = "bold"),
        legend.text = element_text(size = 12)) 

#ggsave("./test/simulate_smalldata_time.jpg", dpi=600)
