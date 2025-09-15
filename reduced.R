setwd("C:/Users/User/Desktop/USP/Projeto/sheets")
library(umap)
library(Rtsne)
library(ggplot2)

data <- read.csv("Metadata_Familia.csv", header = TRUE)

data$Family <- as.factor(data$Family)

data_numerical <- data[, !names(data) %in% c("X", "Family")]

data_numerical[is.na(data_numerical)] <- 0

data_numerical_scaled <- scale(data_numerical)

sum(is.na(data_numerical_scaled))

data_numerical_scaled[is.na(data_numerical_scaled)] <- 0

umap_result <- umap(data_numerical_scaled)

umap_df <- data.frame(UMAP1 = umap_result$layout[,1], UMAP2 = umap_result$layout[,2], Family = data$Family)

ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Family)) +
  geom_point() +
  theme_minimal() +
  labs(title = "UMAP de Dados com Variável Categórica 'Family'",
       x = "UMAP Dimensão 1",
       y = "UMAP Dimensão 2")


library(plotly) 
library(stats) 
data(iris) 
X <- subset(iris, select = -c(Species)) 
axis = list(showline=FALSE, 
            zeroline=FALSE, 
            gridcolor='#ffff', 
            ticklen=4)
fig <- iris %>%  
  plot_ly()  %>%  
  add_trace(  
    type = 'splom',  
    dimensions = list( 
      list(label = 'sepal_width',values=~Sepal.Width),  
      list(label = 'sepal_length',values=~Sepal.Length),  
      list(label ='petal_width',values=~Petal.Width),  
      list(label = 'petal_length',values=~Petal.Length)),  
    color = ~Species, colors = c('#636EFA','#EF553B','#00CC96') 
  ) 
fig <- fig %>% 
  layout( 
    legend=list(title=list(text='species')), 
    hovermode='closest', 
    dragmode= 'select', 
    plot_bgcolor='rgba(240,240,240,0.95)', 
    xaxis=list(domain=NULL, showline=F, zeroline=F, gridcolor='#ffff', ticklen=4), 
    yaxis=list(domain=NULL, showline=F, zeroline=F, gridcolor='#ffff', ticklen=4), 
    xaxis2=axis, 
    xaxis3=axis, 
    xaxis4=axis, 
    yaxis2=axis, 
    yaxis3=axis, 
    yaxis4=axis 
  ) 
fig

library(tsne)
library(plotly)
data("iris")

features <- subset(data, select = -c(Family)) 

set.seed(0)
tsne <- tsne(data_numerical_scaled, initial_dims = 2)
tsne <- data.frame(tsne)
pdb <- cbind(tsne,data$Family)
options(warn = -1)
fig <-  plot_ly(data = pdb ,x =  ~X1, y = ~X2, type = 'scatter', mode = 'markers', split = ~data$Family)

fig <- fig %>%
  layout(
    plot_bgcolor = "#e5ecf6"
  )

fig

tsne_result <- tsne(data_numerical_scaled, initial_dims = 2)
tsne_df <- data.frame(tsne_result)
tsne_df$Family <- data$Family  # Adicionando a coluna da família

# Criando o gráfico
p <- ggplot(tsne_df, aes(x = X1, y = X2, color = Family)) +
  geom_point(alpha = 0.7) +  # Ajuste de transparência para melhor visualização
  theme_minimal() + 
  labs(title = "t-SNE Visualization", x = "t-SNE 1", y = "t-SNE 2")

# Exibir o gráfico
print(p)

# Salvar a imagem
ggsave("tsne_plot.jpeg", plot = p, width = 8, height = 6, dpi = 300)
