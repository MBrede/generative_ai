library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)


plot_a_matrix <- function(plot,
                          matrix,
                          x_offset = 0,
                          y_offset = 0,
                          bracket_kink_length = .05,
                          show_names = F) {
  values <- matrix %>%
    as.data.frame() %>%
    rownames_to_column("row") %>%
    pivot_longer(-row, names_to = "col", values_to = "value") %>%
    mutate(
      y_coord = -1 * scale(as.numeric(factor(row, levels = unique(
        row
      ))), scale = F),
      x_coord = -1 * scale(as.numeric(as.factor(col)), scale = F) + x_offset,
      value = round(value, 2)
    )
  
  brackets <- tibble(
    x = bracket_kink_length * c(-0.5, -1, -1, .5, 1, 1) + rep(range(values$x_coord), each = 3),
    x_end = bracket_kink_length * c(-1, -1, -.5, 1, 1, .5) + rep(range(values$x_coord), each = 3),
    y = bracket_kink_length * c(-1, -1, 1, -1, -1, 1) + rep(rep(range(values$y_coord), times = c(2, 1)), 2),
    y_end = bracket_kink_length * c(-1, 1, 1, -1, 1, 1) + rep(rep(range(values$y_coord), times = c(1, 2)), 2)
  )
  
  plot <- plot +
    geom_point(
      data = values,
      aes(
        x = x_coord,
        y = y_coord,
        size = abs(value),
        color = value
      ),
      alpha = .75
    ) +
    geom_segment(data = brackets, aes(x, y, xend = x_end, yend = y_end)) +
    geom_text(data = values, aes(
      x = x_coord,
      y = y_coord,
      label = round(value, 2)
    ))
  if (show_names) {
    plot <- plot +
      geom_text(
        data = group_by(values, row) |> summarise(
          y_coord = first(y_coord),
          x_coord = min(x_coord) - 2 * bracket_kink_length
        ),
        aes(x = x_coord, y = y_coord, label = row)
      ) +
      geom_text(
        data = group_by(values, col) |> summarise(
          y_coord = max(y_coord) + 2 * bracket_kink_length,
          x_coord = first(x_coord)
        ),
        aes(x = x_coord, y = y_coord, label = col)
      )
  }
  plot
}


plot_an_equation <- function(operation_list,
                             bracket_kink_length = .05,
                             element_offset = 0,
                             show_names = F) {
  x_widths <- sapply(operation_list, \(x) {
    if (is.matrix(x)) {
      ncol(x)
    } else{
      0
    }
  })
  
  x_centers = cumsum(x_widths / 2 + 0.5) +
    cumsum(lag(x_widths / 2, 1, 0)) +
    cumsum(lag(rep(element_offset / 2, length(x_widths)), 1, 0))
  
  
  
  y_heights <- sapply(operation_list, \(x) {
    if (is.matrix(x)) {
      nrow(x)
    } else{
      1
    }
  })
  y_centers <- y_heights / 2 - 0.5
  
  g <- ggplot(data.frame(
    x = c(0, max(x_centers + x_widths)),
    y = c(-.75, .75) * max(y_heights)
  ), aes(x, y))
  
  for (i in seq_along(operation_list)) {
    element = operation_list[[i]]
    y = y_centers[i]
    x = x_centers[i]
    if (is.matrix(element)) {
      g <-  plot_a_matrix(
        plot = g,
        matrix = element,
        x_offset = x,
        y_offset = y,
        bracket_kink_length = bracket_kink_length
      )
    } else{
      g <- g + annotate('text',
                        x = x,
                        y = y,
                        label = element)
    }
  }
  g +
    coord_equal() +
    theme_void() +
    theme(legend.position = 'none') +
    scale_size(range = c(0, 7.5)) +
    scale_color_gradient2(low = '#376795',
                          mid = '#fff',
                          high = '#EF8A47')
}

display_a_multiplication <- function(matrix_a,
                                     matrix_b,
                                     bracket_kink_length = .1,
                                     element_offset = 0,
                                     show_names = F) {
  matrix_c <- matrix_a %*% matrix_b
  rownames(matrix_c) <- rownames(matrix_a)
  colnames(matrix_c) <- colnames(matrix_b)
  
  
  operation_list <- list(matrix_a, 'x', matrix_b, '=', matrix_c)
  plot_an_equation(operation_list,
                   bracket_kink_length = bracket_kink_length,
                   element_offset = element_offset,
                   show_names = show_names)
}


softmax <- function(mat){
    exp(mat)/rowSums(exp(mat))
}

# matrix_a  <-  matrix(c(1, 0, 1, 0, 1), nrow = 1)
# matrix_b <- matrix(c(1, 0, 1, 1, 0), nrow = 5)
# 
# display_a_multiplication(matrix_a,
#                          matrix_b,
#                          bracket_kink_length = 0.2,
#                          element_offset = 0.3)
