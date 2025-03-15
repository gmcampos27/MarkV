# Bibliotecas
library(ggplot2)
library(dplyr)
library(maps)
library(viridis)
library(wesanderson)
library(ggsci)

#Diretório

setwd("path")

# Dados
dados <- read.csv("Family_Metadata.csv")

dados$Country[dados$Country == 'United Kingdom'] <- 'UK'
dados$Country[dados$Country == 'Viet Nam'] <- 'Vietnam'

contagem_pais <- dados %>%
  group_by(Country) %>%
  summarise(quantidade = n())

mapa <- map_data("world")
mapa <- mapa %>% filter(region != "Antarctica")

# Unir os dados de contagem com o mapa
mapa_contagem <- mapa %>%
  left_join(contagem_pais, by = c("region" = "Country"))

pal <- wes_palette("Zissou1", type = "continuous")

ggplot(mapa_contagem, aes(x = long, y = lat, group = group, fill = log(quantidade))) +
  geom_polygon(color = "black") +
  scale_fill_gradientn(colours = pal) +
  theme_void() +
  labs(title = "Distribuição Sequências Virais por País") +
  theme(legend.position = "bottom") +
  xlab("") + ylab("") + labs(fill = "Quantidade de Sequências (log)") + theme(legend.title = element_text(size = 12), legend.text = element_text(size = 10))


contagem_brasil <- dados %>%
  filter(Country == "Brazil") %>%  # Filtra apenas registros do Brasil
  group_by(Family) %>%  # Agrupa por família viral
  summarise(quantidade = n(), .groups = 'drop') %>%  # Conta ocorrências
  arrange(desc(quantidade))  # Ordena da mais frequente para a menos frequente

# Plotar a distribuição das famílias virais no Brasil
ggplot(contagem_brasil, aes(x = quantidade, y = reorder(Family, -quantidade), fill = Family)) +
  geom_bar(stat = "identity") +
  labs(title = "Famílias Virais Brasil",
       x = "Quantidade NCBI Virus",
       y = "Família") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_color_npg()
