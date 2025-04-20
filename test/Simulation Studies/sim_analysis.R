load("./test/Simulation Studies/res_binomial.Rda")
load("./test/Simulation Studies/res_gaussian.Rda")
load("./test/Simulation Studies/res_poisson.Rda")
load("./test/Simulation Studies/res_gamma.Rda")

library(ggplot2)
library(dplyr)
library(tidyr)

wide_df_poisson<-wide_df_poisson[wide_df_poisson$adadelta_MSE<25,]
#wide_df_gamma<-wide_df_gamma[wide_df_gamma$adadelta_MSE<25,]

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

mse_plot <- ggplot(df_long[df_long$metrics=="MSE", ], aes(x = factor(family), y = y_value, fill=Method)) +  
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.7) +
  facet_wrap(~family, scales = "free", nrow=1) +
  guides(fill = guide_legend(byrow = TRUE, nrow = 1)) +
  labs(y = "MSE", fill = "Method", 
       x = "") +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        plot.title = element_text(hjust=0.5, size=15),
        axis.title = element_text(size = 12),
        strip.text = element_text(size = 12, face = "bold")) 

#ggsave("./test/Simulation Studies/simulate_sparse.jpg", dpi=600)

time_plot <- ggplot(df_long[df_long$metrics=="Time", ], aes(x = factor(family), y = y_value, fill=Method)) +  
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.7) +
  facet_wrap(~family, scales = "free", nrow=1) +
  guides(fill = guide_legend(byrow = TRUE, nrow = 1)) +
  labs(y = "Time (ms)", fill = "Method", 
       x = "") +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        plot.title = element_text(hjust=0.5, size=15),
        axis.title = element_text(size = 12),
        strip.text = element_text(size = 12, face = "bold")) 

#ggsave("./test/Simulation Studies/simulate_sparse_time.jpg", dpi=600)

mse_nolegend <- mse_plot + theme(legend.position = "none")
time_nolegend <- time_plot + theme(legend.position = "none")

combined_plot <- plot_grid(
  mse_nolegend,
  time_nolegend,
  ncol = 1,
  align = "v",
  rel_heights = c(1, 1))
combined_plot

legend <- get_legend(mse_plot + theme(legend.position = "bottom", 
                                      legend.text=element_text(size=15), 
                                      legend.title=element_text(size=15)))

final_plot <- plot_grid(
  combined_plot,
  legend,
  ncol = 1,
  rel_heights = c(1, 0.1)  
)

final_plot

#ggsave("./test/Simulation Studies/combinedplot_sparsedata.jpg", dpi=600, width=10, height=10)


