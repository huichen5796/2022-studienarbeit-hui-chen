import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SeiteNnComponent } from './seite-nn.component';

describe('SeiteNnComponent', () => {
  let component: SeiteNnComponent;
  let fixture: ComponentFixture<SeiteNnComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [SeiteNnComponent]
    });
    fixture = TestBed.createComponent(SeiteNnComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
