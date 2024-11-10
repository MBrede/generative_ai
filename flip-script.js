<script>
var elements = document.getElementsByClassName('card');

var myFunction = function() {
  var overlay = this.querySelector('.overlay');
  var content = this.querySelector('.content');
  content.classList.toggle('blur-effect');
  if (overlay) {
    overlay.classList.toggle('show-overlay');
  }
}

for (var i = 0; i < elements.length; i++) {
    elements[i].addEventListener('click', myFunction, false);
    myFunction.call(elements[i]);
}

</script>