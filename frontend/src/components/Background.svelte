<script>
  import { onMount, onDestroy } from 'svelte';

  let vantaContainer;
  let vantaEffect;

  onMount(() => {
    const initVanta = () => {
      if (window.VANTA) {
        try {
          // Dark Theme Configuration for BIRDS
          vantaEffect = window.VANTA.BIRDS({
            el: vantaContainer,
            mouseControls: true,
            touchControls: true,
            gyroControls: false,
            minHeight: 200.00,
            minWidth: 200.00,
            scale: 1.00,
            scaleMobile: 1.00,
            
            // Custom Colors matching KingfisherBot
            backgroundColor: 0x0b0d10, // Matching --bg
            color1: 0x06d6a0,          // Matching --success/accent Green
            color2: 0x4cc9f0,          // Matching --accent Cyan
            colorMode: "lerpGradient",
            
            // Bird Physics - adjusted for "Data Flow" feel
            birdSize: 1.5,
            wingSpan: 20.0,
            speedLimit: 4.0,
            separation: 60.0,
            alignment: 20.0,
            cohesion: 20.0,
            quantity: 3.0 // Moderate density
          });
        } catch (e) {
          console.error("Vanta initialization failed:", e);
        }
      }
    };

    // Retry just in case scripts are loading
    if (!window.VANTA) {
        const interval = setInterval(() => {
            if (window.VANTA) {
                initVanta();
                clearInterval(interval);
            }
        }, 100);
        setTimeout(() => clearInterval(interval), 5000);
    } else {
        initVanta();
    }
  });

  onDestroy(() => {
    if (vantaEffect) vantaEffect.destroy();
  });
</script>

<div bind:this={vantaContainer} class="vanta-bg"></div>

<style>
  .vanta-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0; /* Behind everything */
    pointer-events: none; /* Crucial: allows clicking app elements */
  }
</style>