
include_gif <- function(path, label='', cap=''){
    png_path = gsub('gif', 'png', path)
    counter <- 0
    while(T){
        bash_call <- paste0("convert '", path, "[", counter, "]' ", png_path)
        system(bash_call)
        img <- png::readPNG(png_path)
        if(var(as.vector(img)) > 0){
            break
        }
        counter <- counter + 1
    }
    cat(paste0('![', cap, '](', png_path, '){#',
               label,' .enlarge-onhover .gif-image width=100}'))
}
