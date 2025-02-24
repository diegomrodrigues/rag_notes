from pathlib import Path
from typing import Dict, Any
from ceviche.core.agent import Agent
from ceviche.core.context import Context
import fitz  # PyMuPDF for PDF processing

class ImageProcessorAgent(Agent):
    """Processes images within directory structures and generates metadata."""
    
    def __init__(self, debug: bool = False):
        super().__init__(debug)
        self.debug = debug

    def pre_execution(self, ctx: Context, args: Dict[str, Any]):
        """Prepare context for image processing."""
        if self.debug:
            print("ImageProcessor: pre_execution")
        ctx["base_dir"] = args.get("directory", ".")
        ctx["excluded_dirs"] = args.get("excluded_folders", [])
        ctx["debug"] = self.debug

    def execute(self, ctx: Context, args: Dict[str, Any]) -> Any:
        """Main execution method for image processing."""
        base_dir = Path(ctx["base_dir"])
        excluded_dirs = ctx["excluded_dirs"]
        
        try:
            if self.debug:
                print(f"Starting image processing in: {base_dir}")
            
            # Check if images directory exists and create if needed
            images_dir = base_dir / "images"
            if not images_dir.exists():
                if self.debug:
                    print("Images directory not found. Creating and extracting from PDF...")
                images_dir.mkdir(exist_ok=True)
                pdf_path = self._find_pdf_file(base_dir)
                if pdf_path:
                    self._extract_images_from_pdf(pdf_path, images_dir)
                else:
                    print("⚠️ No PDF file found to extract images from")
            
            # Get the process_images workflow
            process_images_workflow = self.get_workflow("process_images", ctx, args)
            
            # Run the workflow with directory context
            workflow_args = {
                "base_dir": str(base_dir),
                "excluded_dirs": excluded_dirs,
                "pdf_file": self._find_pdf_file(base_dir)
            }
            
            process_images_workflow.run(ctx, workflow_args)
            
            if self.debug:
                print("Image processing completed successfully")
                
        except Exception as e:
            print(f"❌ Image processing failed: {str(e)}")
            raise

    def _extract_images_from_pdf(self, pdf_path: str, output_dir: Path) -> None:
        """Extract all images from PDF and save them to the output directory."""
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    image_name = f"page_{page_num + 1}_img_{img_idx + 1}.{image_ext}"
                    image_path = output_dir / image_name
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                        
            if self.debug:
                print(f"Successfully extracted images from PDF: {pdf_path}")
                
        except Exception as e:
            print(f"❌ Failed to extract images from PDF: {str(e)}")
            raise

    def _find_pdf_file(self, directory: Path) -> str:
        """Find first PDF file in directory for context."""
        pdf_files = list(directory.glob("*.pdf"))
        if pdf_files:
            return str(pdf_files[0])
        return ""

    def post_execution(self, ctx: Context, args: Dict[str, Any], result: Any):
        """Cleanup after processing."""
        if self.debug:
            print("ImageProcessor: post_execution") 