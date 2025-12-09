<script>
var elements = document.getElementsByClassName('card');

var myFunction = function() {
  var overlay = this.querySelector('.overlay');
  var content = this.querySelector('.content');
  content.classList.toggle('blur-effect');
  if (overlay) {
    overlay.classList.toggle('show-overlay')
  }
}

for (var i = 0; i < elements.length; i++) {
    elements[i].addEventListener('click', myFunction, false);
    myFunction.call(elements[i]);
}

document.addEventListener('DOMContentLoaded', function() {
  const images = document.querySelectorAll('.gif-image');
  
  images.forEach(function(image) {
    image.addEventListener('click', function() {
        console.log(this.src)
        console.log(this.src.slice(0,-4))
        if(this.src.substr(-4) == '.gif') {
          this.src = this.src.slice(0,-4) + '.png'
        } else {
          this.src = this.src.slice(0,-4) + '.gif'
        }
      });
  });
});


document.querySelectorAll('.enlarge-onhover').forEach(el => {
  el.addEventListener('click', e => {
    e.stopPropagation();
    el.classList.toggle('active');
  });
});

document.addEventListener('click', () => {
  document.querySelectorAll('.enlarge-onhover.active').forEach(el => el.classList.remove('active'));
});


</script>