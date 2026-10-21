const menuToggle = document.querySelector(".menu-toggle");
const nav = document.querySelector(".nav");

if (menuToggle) {
  menuToggle.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    menuToggle.setAttribute("aria-expanded", String(open));
  });
}

document.querySelectorAll(".nav a").forEach(link => {
  link.addEventListener("click", () => {
    nav.classList.remove("open");
    menuToggle?.setAttribute("aria-expanded", "false");
  });
});

// Pequena confirmação visual ao passar pelos cards de resultado.
document.querySelectorAll(".model-card").forEach(card => {
  card.addEventListener("mouseenter", () => card.classList.add("active"));
  card.addEventListener("mouseleave", () => card.classList.remove("active"));
});
